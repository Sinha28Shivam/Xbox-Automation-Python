"""
builder.py - construct the LangGraph workflow from graph.yaml.

There is no hand-written graph anywhere in this codebase. This module reads the
YAML and calls add_node / add_edge / add_conditional_edges accordingly, so
re-wiring the workflow - inserting an agent, changing where failures go, adding
a retry loop - is a config edit.

That is not architectural purity for its own sake. Test workflows change often
and per-team: one site wants recovery attempts, another wants RCA before every
retry, a third wants to skip planning for literal step lists. Encoding those as
edges in Python would mean a fork per team.

DISABLED AGENTS AND EDGE HEALING
--------------------------------
An agent with `enabled: false` is skipped and its edges are repaired, so the
predecessor connects to the successor. Without that, turning off `recovery`
would leave graph.yaml pointing at a node that does not exist. Toggling an
agent is therefore genuinely one line, with no follow-up edits.
"""

from __future__ import annotations

import importlib
from typing import Any, Callable

from config import Config
from state import AgenticState

try:
    from langgraph.graph import END, START, StateGraph
except ImportError as exc:                            # pragma: no cover
    raise SystemExit(
        "LangGraph is required:  pip install langgraph\n"
        f"  ({exc})")


class GraphBuildError(RuntimeError):
    pass


def _resolve(dotted: str) -> Any:
    """Import 'module.attribute' at runtime.

    Used for both router functions and agent classes, which is what lets
    agents.yaml and graph.yaml name implementations without this module
    importing any of them directly.
    """
    if "." not in dotted:
        raise GraphBuildError(
            f"'{dotted}' must be a dotted path like 'module.function'")
    module_name, _, attr = dotted.rpartition(".")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise GraphBuildError(
            f"Could not import '{module_name}' for '{dotted}': {exc}") from exc
    if not hasattr(module, attr):
        raise GraphBuildError(f"'{module_name}' has no attribute '{attr}'")
    return getattr(module, attr)


class WorkflowBuilder:
    """Turns graph.yaml + agents.yaml into a compiled LangGraph app."""

    def __init__(self, graph_config: Config, agents_config: Config,
                 agent_factory: Callable[[str], Any]):
        self.graph_config = graph_config
        self.agents_config = agents_config
        self.agent_factory = agent_factory

        self.route_mode = str(graph_config.get("route_mode", "static"))
        self.entry_point = str(graph_config.get("entry_point", ""))
        if not self.entry_point:
            raise GraphBuildError("graph.yaml must declare an entry_point")

        self.enabled: set[str] = {
            role for role, spec in (agents_config.section("agents") or {}).items()
            if spec.get("enabled", True)
        }

    # -- build -------------------------------------------------------------
    def build(self) -> Any:
        graph = StateGraph(AgenticState)

        nodes = [n for n in self.graph_config.list_of("nodes")
                 if str(n) in self.enabled]
        if not nodes:
            raise GraphBuildError(
                "No enabled nodes. Every agent in agents.yaml is disabled.")

        for node in nodes:
            graph.add_node(str(node), self.agent_factory(str(node)))

        entry = self._first_enabled(self.entry_point)
        if entry is None:
            raise GraphBuildError(
                f"Entry point '{self.entry_point}' is disabled and no enabled "
                f"successor was found.")
        graph.add_edge(START, entry)

        if self.route_mode == "supervised":
            self._wire_supervised(graph, nodes)
        else:
            self._wire_static(graph, nodes)

        return graph.compile(checkpointer=self._checkpointer())

    # -- static wiring -----------------------------------------------------
    def _wire_static(self, graph: Any, nodes: list[str]) -> None:
        wired: set[str] = set()

        for edge in self.graph_config.list_of("edges"):
            source, target = str(edge.get("from")), str(edge.get("to"))
            if source not in self.enabled:
                continue
            resolved = END if target == "__end__" else self._first_enabled(target)
            if resolved is None:
                resolved = END
            graph.add_edge(source, resolved)
            wired.add(source)

        for spec in self.graph_config.list_of("conditional_edges"):
            source = str(spec.get("from"))
            if source not in self.enabled:
                continue

            router = _resolve(str(spec.get("router")))
            mapping: dict[str, Any] = {}
            for condition, target in (spec.get("map") or {}).items():
                resolved = (END if target == "__end__"
                            else self._first_enabled(str(target)))
                # A route to a disabled node falls through to END rather than
                # crashing at compile time. Combined with edge healing, this
                # keeps the graph valid under any enable/disable combination.
                mapping[str(condition)] = resolved if resolved else END

            graph.add_conditional_edges(source, router, mapping)
            wired.add(source)

        # Any node with no outgoing edge would hang the graph. Terminating at
        # END is the safe default - the reporter has usually already run.
        for node in nodes:
            if node not in wired:
                graph.add_edge(node, END)

    # -- supervised wiring -------------------------------------------------
    def _wire_supervised(self, graph: Any, nodes: list[str]) -> None:
        """Every node returns to the supervisor, which picks what runs next."""
        if "supervisor" not in self.enabled:
            raise GraphBuildError(
                "route_mode is 'supervised' but the supervisor agent is "
                "disabled in agents.yaml.")

        from routing import route_supervised

        workers = [n for n in nodes if n != "supervisor"]
        mapping: dict[str, Any] = {n: n for n in workers}
        mapping["__end__"] = END
        graph.add_conditional_edges("supervisor", route_supervised, mapping)

        for node in workers:
            graph.add_edge(node, "supervisor")

    # -- helpers -----------------------------------------------------------
    def _first_enabled(self, start: str) -> str | None:
        """`start` if enabled, else follow its edges to the first enabled node.

        This is the edge healing. When `recovery` is off, an edge pointing at
        it resolves to whatever recovery itself pointed to.
        """
        if start in self.enabled:
            return start

        seen: set[str] = set()
        frontier = [start]
        while frontier:
            current = frontier.pop(0)
            if current in seen:
                continue
            seen.add(current)

            for edge in self.graph_config.list_of("edges"):
                if str(edge.get("from")) == current:
                    target = str(edge.get("to"))
                    if target == "__end__":
                        return None
                    if target in self.enabled:
                        return target
                    frontier.append(target)

            for spec in self.graph_config.list_of("conditional_edges"):
                if str(spec.get("from")) == current:
                    for target in (spec.get("map") or {}).values():
                        if str(target) in self.enabled:
                            return str(target)
        return None

    def _checkpointer(self) -> Any:
        """Persist state between nodes when configured.

        Beyond resumability, the real value is forensic: the checkpoint holds
        exactly what each agent saw, which turns "why did the verifier decide
        that?" into something you can inspect instead of guess at.
        """
        spec = self.graph_config.section("checkpointer")
        if not spec.get("enabled", False):
            return None

        kind = str(spec.get("type", "memory"))
        try:
            if kind == "sqlite":
                from langgraph.checkpoint.sqlite import SqliteSaver
                return SqliteSaver.from_conn_string(
                    str(spec.get("sqlite_path", "./checkpoints.sqlite")))
            from langgraph.checkpoint.memory import MemorySaver
            return MemorySaver()
        except ImportError:
            # Checkpointing is a convenience; losing it must not stop a run.
            return None


def build_workflow(graph_config: Config, agents_config: Config,
                   agent_factory: Callable[[str], Any]) -> Any:
    return WorkflowBuilder(graph_config, agents_config, agent_factory).build()
