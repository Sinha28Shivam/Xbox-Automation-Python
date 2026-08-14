"""
registry.py - the tool catalogue agents draw from.

HOW TOOLS ARE GRANTED
---------------------
agents.yaml lists what each agent may call, either by name or by tag:

    tools: ["tag:vision", "press_button"]

The registry resolves those at bind time. An agent physically cannot call a
tool it was not granted, which is a real safety property here: the verifier
must not be able to press buttons, or it could "fix" a failing test by nudging
the console and then declare success. Separation of powers, enforced by
construction rather than by instruction.

WHY A CUSTOM REGISTRY RATHER THAN @tool EVERYWHERE
--------------------------------------------------
Every tool needs the same three things injected: the hardware bridge, the
artifact store and the runtime config. LangChain's @tool decorator wants plain
functions. So tools are declared as factories taking a ToolContext and
returning LangChain StructuredTools; the registry does the wiring, and each
tool ends up with hardware access without any global state.

Tools return a JSON-serialisable dict, always including "ok". Agents read the
result, so a raised exception would abort the whole node - instead failures come
back as {"ok": false, "error": ...} and the agent decides what to do.
"""

from __future__ import annotations

import difflib
import importlib
import inspect
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from adapters import HardwareBridge
from artifacts import ArtifactStore
from config import Config


@dataclass
class ToolContext:
    """Everything a tool needs. Passed to every tool factory."""

    hardware: HardwareBridge
    artifacts: ArtifactStore
    settings: Config
    dry_run: bool = False
    # Scratch space shared across tool calls within one run - used mainly to
    # keep the last frame so a "did the screen change?" check has a baseline
    # without re-grabbing.
    scratch: dict[str, Any] = field(default_factory=dict)

    def threshold(self, key: str, fallback: float) -> float:
        return self.settings.get(f"verification.{key}", fallback)


@dataclass
class ToolSpec:
    """One registered tool."""

    name: str
    description: str
    tags: list[str]
    factory: Callable[[ToolContext], Any]
    # Tools that touch hardware are skipped in dry-run mode; read-only ones
    # (frame stats, config introspection) still work, so a plan can be
    # rehearsed end to end without a console attached.
    mutates_hardware: bool = False


class ToolRegistry:
    """Holds ToolSpecs and builds the concrete tools for an agent."""

    def __init__(self, context: ToolContext):
        self.context = context
        self._specs: dict[str, ToolSpec] = {}
        self._built: dict[str, Any] = {}

    # -- registration ------------------------------------------------------
    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._specs:
            raise ValueError(f"Duplicate tool name '{spec.name}'")
        self._specs[spec.name] = spec

    def register_many(self, specs: list[ToolSpec]) -> None:
        for s in specs:
            self.register(s)

    def load_modules(self, module_names: list[str]) -> None:
        """Import tool modules and call their `provide()` function.

        Each module exposes `provide() -> list[ToolSpec]`. Adding a tool module
        is therefore a config entry, keeping the registry itself closed to
        modification as new capabilities appear.
        """
        for name in module_names:
            try:
                module = importlib.import_module(name)
            except ImportError as exc:
                raise ImportError(
                    f"Tool module '{name}' could not be imported: {exc}") from exc
            provide = getattr(module, "provide", None)
            if provide is None:
                raise AttributeError(
                    f"Tool module '{name}' must define provide() -> list[ToolSpec]")
            self.register_many(provide())

    # -- lookup ------------------------------------------------------------
    @property
    def names(self) -> list[str]:
        return sorted(self._specs)

    def spec(self, name: str) -> ToolSpec:
        if name not in self._specs:
            raise KeyError(
                f"Unknown tool '{name}'. Registered: {', '.join(self.names)}")
        return self._specs[name]

    def by_tag(self, tag: str) -> list[ToolSpec]:
        return [s for s in self._specs.values() if tag in s.tags]

    def resolve(self, selectors: list[str]) -> list[ToolSpec]:
        """Turn ["tag:vision", "press_button"] into concrete specs.

        Order is preserved and duplicates removed, so an agent granted both a
        tag and a specific tool inside it does not receive it twice.
        """
        out: list[ToolSpec] = []
        seen: set[str] = set()
        for sel in selectors or []:
            sel = str(sel).strip()
            matches = (self.by_tag(sel[4:]) if sel.startswith("tag:")
                       else [self.spec(sel)])
            for spec in matches:
                if spec.name not in seen:
                    seen.add(spec.name)
                    out.append(spec)
        return out

    # -- building ----------------------------------------------------------
    def build(self, name: str) -> Any:
        """Instantiate one tool (cached - tools are stateless per run)."""
        if name not in self._built:
            self._built[name] = self.spec(name).factory(self.context)
        return self._built[name]

    def build_for(self, selectors: list[str]) -> list[Any]:
        """The concrete tool objects an agent is allowed to use."""
        tools: list[Any] = []
        for spec in self.resolve(selectors):
            if self.context.dry_run and spec.mutates_hardware:
                # Skipped rather than faked: an agent that believes it pressed
                # a button in dry-run would produce a plausible but meaningless
                # transcript.
                continue
            tools.append(self.build(spec.name))
        return tools

    def describe(self, selectors: list[str] | None = None) -> list[dict[str, Any]]:
        """Human/LLM-readable catalogue, for prompts and `--list-tools`."""
        specs = (self.resolve(selectors) if selectors
                 else sorted(self._specs.values(), key=lambda s: s.name))
        return [
            {
                "name": s.name,
                "description": s.description.strip(),
                "tags": s.tags,
                "mutates_hardware": s.mutates_hardware,
            }
            for s in specs
        ]


# ===========================================================================
# Helpers used by every tool module
# ===========================================================================
def ok(**payload: Any) -> dict[str, Any]:
    """Successful tool result."""
    return {"ok": True, **payload}


def fail(error: str, **payload: Any) -> dict[str, Any]:
    """Failed tool result.

    Returned, not raised: the agent needs to see the failure and react (retry,
    replan, or record it as evidence). An exception would kill the node and
    lose that context.
    """
    return {"ok": False, "error": error, **payload}


def tolerant(func: Callable[..., Any], name: str) -> Callable[..., Any]:
    """Wrap a tool so a near-miss argument name does not abort the run.

    WHY THIS EXISTS
    ---------------
    A real failure: the planner emitted
    `wait_for_stable_screen(timeout=10.0, stability_duration=1.0)`. The real
    parameter is `settle`, so Python raised TypeError, the executor treated it
    as a step failure, and a 6-step plan died at step 2 with the console
    already mid-transition.

    That is a disproportionate punishment for a synonym. The LLM's intent was
    perfectly clear, and losing three-quarters of a hardware run over a
    parameter alias is not a useful kind of strictness.

    So we:
      1. rename known aliases to the real parameter,
      2. drop genuinely unknown arguments, recording them in the result,
      3. still fail loudly for a MISSING REQUIRED argument, because that is a
         real ambiguity we must not paper over.

    Dropped arguments are reported, not hidden - the report shows exactly what
    was ignored, so a persistent mismatch is visible rather than silent.
    """
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):                   # pragma: no cover
        return func

    accepted = set(signature.parameters)
    # Synonyms models reach for naturally. Cheaper to accept than to fight.
    aliases = {
        "stability_duration": "settle",
        "stable_duration": "settle",
        "settle_time": "settle",
        "timeout_seconds": "timeout",
        "max_wait": "timeout",
        "seconds": "timeout",
        "frame1": "path_a",
        "frame2": "path_b",
        "frame_a": "path_a",
        "frame_b": "path_b",
        "before": "before_path",
        "after": "after_path",
        "repeat": "times",
        "count": "times",
        "button_name": "button",
        "control": "button",
        "query": "text",
        "search_text": "text",
        "path": "frame_path",
        "image_path": "frame_path",
    }

    def wrapper(**kwargs: Any) -> dict[str, Any]:
        cleaned: dict[str, Any] = {}
        renamed: dict[str, str] = {}
        dropped: dict[str, Any] = {}

        for key, value in kwargs.items():
            if key in accepted:
                cleaned[key] = value
                continue

            target = aliases.get(key)
            if target and target in accepted and target not in kwargs:
                cleaned[target] = value
                renamed[key] = target
                continue

            close = difflib.get_close_matches(key, accepted, n=1, cutoff=0.75)
            if close and close[0] not in kwargs:
                cleaned[close[0]] = value
                renamed[key] = close[0]
                continue

            dropped[key] = value

        # A missing REQUIRED argument is a genuine error - guessing a value
        # would be worse than failing, because it would run the wrong action.
        missing = [
            p.name for p in signature.parameters.values()
            if p.default is inspect.Parameter.empty
            and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
            and p.name not in cleaned
        ]
        if missing:
            return fail(
                f"{name} is missing required argument(s): {', '.join(missing)}. "
                f"Accepted arguments: {', '.join(sorted(accepted)) or 'none'}",
                dispatched=False)

        result = func(**cleaned)

        if isinstance(result, dict) and (renamed or dropped):
            notes = []
            if renamed:
                notes.append("renamed " + ", ".join(
                    f"{k}->{v}" for k, v in renamed.items()))
            if dropped:
                notes.append(f"ignored unknown: {', '.join(dropped)}")
            result["argument_notes"] = "; ".join(notes)
            result["accepted_arguments"] = sorted(accepted)
        return result

    # Copy the ORIGINAL function's metadata onto the wrapper. LangChain builds
    # its argument schema with typing.get_type_hints(), which reads
    # __annotations__ - not __signature__. Setting only the signature left the
    # annotations empty and raised KeyError: 'frame_path' at tool-build time.
    # Both must be carried across for the tool to be introspectable.
    wrapper.__name__ = name
    wrapper.__qualname__ = name
    wrapper.__doc__ = func.__doc__
    wrapper.__annotations__ = dict(getattr(func, "__annotations__", {}))
    wrapper.__signature__ = signature                 # type: ignore[attr-defined]
    # get_type_hints() resolves string annotations against the defining
    # module's globals, so the wrapper must claim the same module.
    wrapper.__module__ = getattr(func, "__module__", wrapper.__module__)
    if hasattr(func, "__globals__"):
        wrapper.__wrapped__ = func                    # type: ignore[attr-defined]
    return wrapper


def make_tool(func: Callable[..., Any], name: str, description: str,
              args_schema: Any = None) -> Any:
    """Wrap a plain function as a LangChain StructuredTool.

    The function is made argument-tolerant first, so both LLM tool-calls and
    direct `call_tool()` invocations survive a plausible parameter alias.

    Falls back to returning the function itself if LangChain is not installed,
    which keeps the tools unit-testable without the LLM stack.
    """
    func = tolerant(func, name)

    # The real signature is appended to the description so the planner and any
    # tool-calling model can see the exact parameter names. Aliases are a
    # safety net; telling the model the truth up front is the actual fix.
    try:
        signature = inspect.signature(func)
        params = ", ".join(
            p.name if p.default is inspect.Parameter.empty
            else f"{p.name}={p.default!r}"
            for p in signature.parameters.values())
        description = f"{description}\n\nSignature: {name}({params})"
    except (TypeError, ValueError):                   # pragma: no cover
        pass

    try:
        from langchain_core.tools import StructuredTool
    except ImportError:                               # pragma: no cover
        func.__name__ = name                          # type: ignore[attr-defined]
        func.__doc__ = description                    # type: ignore[attr-defined]
        return func

    kwargs: dict[str, Any] = {
        "func": func,
        "name": name,
        "description": description,
    }
    if args_schema is not None:
        kwargs["args_schema"] = args_schema
    return StructuredTool.from_function(**kwargs)


def build_default_registry(hardware: HardwareBridge, artifacts: ArtifactStore,
                           settings: Config,
                           module_names: list[str] | None = None) -> ToolRegistry:
    """Create the registry with the configured tool modules loaded."""
    context = ToolContext(
        hardware=hardware,
        artifacts=artifacts,
        settings=settings,
        dry_run=settings.get("runtime.dry_run", False),
    )
    registry = ToolRegistry(context)
    registry.load_modules(module_names or [
        "hardware_tools", "input_tools", "vision_tools", "report_tools",
    ])
    return registry
