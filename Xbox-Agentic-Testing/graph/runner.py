"""
runner.py - assemble everything and run one scenario.

This is where the pieces meet: config is loaded, adapters are wired, tools are
registered, agents are instantiated from agents.yaml, the graph is built from
graph.yaml, and the whole thing is invoked.

It is also the only place that knows about the framework's directory layout, so
if the repo moves, one file changes.

WHY AGENTS ARE INSTANTIATED BY DOTTED PATH
------------------------------------------
Each agent's `impl` in agents.yaml names its class. The runner imports it at
startup. Replacing an agent - a stricter verifier, a planner tuned for a
different console - is then a config change plus a new file, with nothing here
to edit. That is the extension point that makes the roster genuinely open.

CLEANUP MATTERS HERE
--------------------
The capture device is held exclusively while we run. If the process exits
without releasing it, the next run - and RECentral, and anything else - cannot
open the card. So teardown happens in a finally block, on every path including
Ctrl+C.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

# The framework's own packages are added to sys.path so that modules can import
# each other by bare name (`from config import Config`). Flat imports keep the
# dotted paths in agents.yaml short and readable, which matters because users
# edit that file.
_ROOT = Path(__file__).resolve().parent.parent
for _sub in ("core", "tools", "agents", "graph"):
    _path = str(_ROOT / _sub)
    if _path not in sys.path:
        sys.path.insert(0, _path)

from adapters import HardwareBridge                      # noqa: E402
from artifacts import ArtifactStore, new_run_id          # noqa: E402
from base import AgentContext                            # noqa: E402
from builder import _resolve, build_workflow             # noqa: E402
from config import Config, load_dotenv_if_present        # noqa: E402
from llm import LLMFactory                               # noqa: E402
from prompts import PromptLibrary                        # noqa: E402
from registry import build_default_registry              # noqa: E402
from schemas import TestReport, Verdict                  # noqa: E402
from state import initial_state                          # noqa: E402


CONFIG_FILES = {
    "settings": "settings.yaml",
    "agents": "agents.yaml",
    "graph": "graph.yaml",
}


class TestRunner:
    """Runs scenarios through the multi-agent workflow."""

    def __init__(self, config_dir: Path | str | None = None,
                 overrides: dict[str, Any] | None = None):
        self.root = _ROOT
        self.config_dir = Path(config_dir) if config_dir else self.root / "config"

        # Loaded before any config, because settings.yaml expands ${VAR}
        # references at parse time and needs the environment already populated.
        load_dotenv_if_present(self.root / ".env")
        load_dotenv_if_present(self.root.parent / ".env")

        # base=self.root: relative paths in settings.yaml ("./scenarios",
        # "../Xbox-Automation-Python") are written relative to the PROJECT,
        # not to the config/ folder the file happens to sit in.
        configs = Config.load_all(self.config_dir, CONFIG_FILES, base=self.root)

        self.settings = configs["settings"]
        self.agents_config = configs["agents"]
        self.graph_config = configs["graph"]

        if overrides:
            self.settings = self.settings.merged_with({"runtime": overrides})

        self.hardware = HardwareBridge(self.settings)
        self.llm_factory = LLMFactory(self.settings)
        self.prompts = PromptLibrary(
            self.settings.resolve_path("paths.prompts_dir", "./config/prompts"))

    # -- running -----------------------------------------------------------
    def run(self, scenario_input: str, scenario_source: str = "text",
            run_id: str | None = None) -> TestReport | None:
        run_id = run_id or new_run_id()
        started_at = datetime.now(timezone.utc).isoformat()

        artifacts = ArtifactStore(
            root=self.settings.resolve_path("paths.artifacts_dir", "./artifacts"),
            run_id=run_id,
            frame_format=self.settings.get("runtime.frame_format", "png"),
            enabled=self.settings.get("runtime.save_frames", True),
        )

        tools = build_default_registry(self.hardware, artifacts, self.settings)
        agent_context = AgentContext(
            settings=self.settings,
            agents_config=self.agents_config,
            llm_factory=self.llm_factory,
            prompts=self.prompts,
            tools=tools,
            artifacts=artifacts,
        )

        workflow = build_workflow(
            self.graph_config, self.agents_config,
            self._agent_factory(agent_context))

        state = initial_state(
            run_id=run_id,
            scenario_input=scenario_input,
            scenario_source=scenario_source,
            # The routers read limits from here, which keeps them pure
            # functions of the state instead of reaching for global config.
            config={
                "runtime": self.settings.section("runtime"),
                "agents": self.agents_config.section("agents"),
                "verification": self.settings.section("verification"),
            },
            run_dir=str(artifacts.run_dir),
            dry_run=self.settings.get("runtime.dry_run", False),
            started_at=started_at,
        )

        print(f"\nRun {run_id}")
        print(f"Artifacts: {artifacts.run_dir}\n")

        try:
            final = workflow.invoke(state, config={
                "recursion_limit": self.graph_config.get(
                    "limits.recursion_limit", 150),
                # A thread id is required whenever a checkpointer is attached;
                # using the run id keeps each run's history separate.
                "configurable": {"thread_id": run_id},
            })
        except KeyboardInterrupt:
            print("\nInterrupted - releasing the capture device ...")
            return None
        finally:
            # Always release the card, on every exit path. Leaving it held
            # blocks the next run and every other capture application.
            self.hardware.close()

        return final.get("report")

    def _agent_factory(self, context: AgentContext) -> Callable[[str], Any]:
        """Instantiate an agent from its `impl` dotted path in agents.yaml."""
        def factory(role: str) -> Any:
            spec = context.spec_for(role)
            impl = spec.get("impl")
            if not impl:
                raise ValueError(
                    f"Agent '{role}' has no 'impl' in agents.yaml")
            cls = _resolve(str(impl))
            return cls(context, role)
        return factory

    # -- introspection -----------------------------------------------------
    def describe(self) -> dict[str, Any]:
        """A snapshot of the configured system, for `--info`.

        Handy for diagnosing a run before starting one: which adapters loaded,
        which provider is active, which agents are enabled.
        """
        artifacts = ArtifactStore(
            root=self.settings.resolve_path("paths.artifacts_dir", "./artifacts"),
            run_id="describe", enabled=False)
        tools = build_default_registry(self.hardware, artifacts, self.settings)

        try:
            provider = self.llm_factory.default_provider
            model = self.llm_factory.provider_spec().get("params", {}).get("model")
        except Exception as exc:
            provider, model = f"<error: {exc}>", None

        return {
            "config_dir": str(self.config_dir),
            "llm": {"provider": provider, "model": model},
            "hardware": self.hardware.status(),
            "agents": {
                role: {
                    "enabled": spec.get("enabled", True),
                    "impl": spec.get("impl"),
                    "tools": spec.get("tools", []),
                }
                for role, spec in (self.agents_config.section("agents") or {}).items()
            },
            "tools": tools.describe(),
            "route_mode": self.graph_config.get("route_mode", "static"),
            "runtime": self.settings.section("runtime"),
        }

    def close(self) -> None:
        self.hardware.close()

    def __enter__(self) -> "TestRunner":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def verdict_exit_code(report: TestReport | None) -> int:
    """Map a verdict to a shell exit code.

    BLOCKED gets its own code (2) rather than reusing failure. A CI job that
    cannot tell "the console is broken" from "the rig is broken" will end up
    ignoring both.
    """
    if report is None:
        return 3
    return {
        Verdict.PASS: 0,
        Verdict.FAIL: 1,
        Verdict.BLOCKED: 2,
        Verdict.INCONCLUSIVE: 3,
        Verdict.ERROR: 4,
        Verdict.SKIPPED: 0,
    }.get(report.verdict, 4)
