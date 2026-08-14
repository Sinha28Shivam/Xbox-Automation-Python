"""
_smoke_test.py - verify the framework is wired correctly, without hardware.

Run this after any change to config, agents or the graph:

    python _smoke_test.py

WHAT IT CHECKS
--------------
Everything that can be checked without a console, a capture card or an API key:
config loading, adapter imports, the tool registry, prompt rendering, graph
construction, the routers, and - most importantly - the no-pass-without-proof
invariant in the schemas.

WHY THAT LAST ONE MATTERS MOST
------------------------------
Test 7 asserts that a PASS with no observational evidence is downgraded to
INCONCLUSIVE. That single rule is what stops this framework repeating the
project's original failure: six consecutive "successful" runs while the console
sat doing nothing. If that test ever fails, the framework is dangerous - it can
report green on no evidence - and nothing else here matters until it is fixed.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for sub in ("core", "tools", "agents", "graph"):
    p = str(ROOT / sub)
    if p not in sys.path:
        sys.path.insert(0, p)

PASSED: list[str] = []
FAILED: list[tuple[str, str]] = []


def check(name: str):
    """Decorator: run a check, record the outcome, never abort the suite."""
    def wrap(fn):
        try:
            detail = fn()
            PASSED.append(f"{name}{f' - {detail}' if detail else ''}")
        except Exception as exc:
            FAILED.append((name, f"{exc.__class__.__name__}: {exc}"))
            if "-v" in sys.argv:
                traceback.print_exc()
        return fn
    return wrap


# ===========================================================================
@check("1. config loads with env expansion and type coercion")
def _config():
    from config import Config
    cfg = Config.load(ROOT / "config" / "settings.yaml", base=ROOT)

    max_steps = cfg.get("runtime.max_steps", 40)
    assert isinstance(max_steps, int), f"max_steps should be int, got {type(max_steps)}"

    dry = cfg.get("runtime.dry_run", False)
    assert isinstance(dry, bool), f"dry_run should be bool, got {type(dry)}"

    controls = cfg.resolve_path("paths.controls_config", "")
    assert controls.is_file(), f"controls.yaml not found at {controls}"
    return f"provider={cfg.get('llm.provider', '')}, max_steps={max_steps}"


@check("2. hardware adapters import the existing modules")
def _adapters():
    from adapters import HardwareBridge
    from config import Config

    bridge = HardwareBridge(Config.load(ROOT / "config" / "settings.yaml", base=ROOT))
    status = bridge.status()
    broken = [n for n, a in status["adapters"].items() if not a["available"]]
    assert not broken, f"adapters failed to load: {broken}"

    summary = bridge.controls_summary()
    assert "error" not in summary, summary.get("error")
    assert summary["buttons"], "no buttons found in controls.yaml"
    return f"{len(summary['buttons'])} buttons, {len(summary['macros'])} macros"


@check("3. tool registry builds and enforces tag scoping")
def _tools():
    from artifacts import ArtifactStore
    from adapters import HardwareBridge
    from config import Config
    from registry import build_default_registry

    cfg = Config.load(ROOT / "config" / "settings.yaml", base=ROOT)
    registry = build_default_registry(
        HardwareBridge(cfg),
        ArtifactStore(ROOT / "artifacts", "smoke", enabled=False),
        cfg)

    assert "press_button" in registry.names
    assert "verify_screen_changed" in registry.names

    # The separation of powers that keeps the verifier honest: it can look at
    # frames but must not be able to press buttons and "fix" a failing test.
    verifier_tools = {s.name for s in registry.resolve(["tag:vision", "tag:analysis"])}
    assert "press_button" not in verifier_tools, \
        "SECURITY: the verifier can press buttons - it could fix its own test"
    return f"{len(registry.names)} tools registered"


@check("4. every agent's prompt template renders")
def _prompts():
    from config import Config
    from prompts import PromptLibrary

    agents_cfg = Config.load(ROOT / "config" / "agents.yaml", base=ROOT)
    library = PromptLibrary(ROOT / "config" / "prompts")

    missing = [
        spec["prompt"]
        for spec in (agents_cfg.section("agents") or {}).values()
        if spec.get("prompt") and not library.exists(spec["prompt"])
    ]
    assert not missing, f"prompt templates missing: {missing}"
    return f"{len(agents_cfg.section('agents'))} agents, all templates present"


@check("5. the LangGraph workflow compiles from graph.yaml")
def _graph():
    from builder import build_workflow
    from config import Config

    graph_cfg = Config.load(ROOT / "config" / "graph.yaml", base=ROOT)
    agents_cfg = Config.load(ROOT / "config" / "agents.yaml", base=ROOT)

    # A stub factory: we are testing the topology, not the agents.
    workflow = build_workflow(
        graph_cfg, agents_cfg,
        lambda role: (lambda state: {"messages": [{"role": role, "text": "stub"}]}))
    assert workflow is not None
    return f"entry={graph_cfg.get('entry_point', '')}, mode={graph_cfg.get('route_mode', '')}"


@check("6. routers are pure and handle missing state")
def _routing():
    import routing
    from schemas import ExecutionResult, HealthReport, VerificationResult, Verdict

    # An empty state must not crash and must fail safe.
    assert routing.route_after_health({}) == "blocked"
    assert routing.route_after_executor({}) == "aborted"
    assert routing.route_after_verification({}) == "failed"

    assert routing.route_after_health(
        {"health": HealthReport(healthy=True)}) == "healthy"

    # A run where nothing ever changed on screen must NOT be replanned -
    # retrying against an unresponsive console just repeats the failure.
    state = {
        "verification": VerificationResult(
            scenario_id="x", verdict=Verdict.FAIL, should_replan=True),
        "execution": ExecutionResult(scenario_id="x", steps=[]),
        "config": {"runtime": {"max_replans": 3}},
        "replan_count": 0,
    }
    assert routing.route_after_verification(state) == "failed", \
        "replanned despite the console never responding"
    return "fail-safe defaults verified"


@check("7. CRITICAL: a PASS with no evidence is downgraded")
def _no_false_pass():
    from schemas import Evidence, EvidenceKind, VerificationResult, Verdict

    # Command acknowledgements only - exactly the false-pass scenario.
    result = VerificationResult(
        scenario_id="x",
        verdict=Verdict.PASS,
        summary="Everything worked!",
        evidence=[Evidence(kind=EvidenceKind.COMMAND_ACK,
                           summary="press_button dispatched=True")],
    )
    assert result.verdict == Verdict.INCONCLUSIVE, \
        "FALSE PASS POSSIBLE: a pass was accepted on command acks alone"
    assert not result.passed
    assert result.not_proven, "the downgrade was not explained"

    # With real observational evidence, a pass is allowed.
    good = VerificationResult(
        scenario_id="x",
        verdict=Verdict.PASS,
        evidence=[Evidence(kind=EvidenceKind.SCREEN_DIFF,
                           summary="screen changed, delta 12.4")],
    )
    assert good.verdict == Verdict.PASS and good.passed, \
        "a legitimate evidence-backed pass was rejected"
    return "false passes blocked, real passes allowed"


@check("8. a scenario with no success criteria is rejected")
def _scenario_gate():
    from schemas import ValidatedScenario

    scenario = ValidatedScenario(
        id="x", title="Vague", goal="make sure it works", success_criteria=[])
    assert not scenario.valid, \
        "a scenario with no criteria was accepted - it could never fail"
    return "untestable scenarios rejected"


@check("9. a plan step without an expected observation is rejected")
def _step_gate():
    from pydantic import ValidationError
    from schemas import PlannedStep

    try:
        PlannedStep(index=0, action="press_button", intent="press A",
                    expected_observation="")
    except ValidationError:
        return "unverifiable steps rejected"
    raise AssertionError("a step with no expected observation was accepted")


@check("10. capture device resolves BY NAME, not by index")
def _capture_device():
    """Regression guard for a real bug.

    The OpenCV indices swapped, the framework opened the laptop webcam, and
    reported "black screen / no HDMI signal" while the Xbox was fine on the
    other index. Name-based resolution is what prevents a repeat, so this
    asserts the mechanism is present and actually finds the card.
    """
    import sys
    from config import Config

    cfg = Config.load(ROOT / "config" / "settings.yaml", base=ROOT)
    automation = cfg.resolve_path("paths.automation_root", "")
    sys.path.insert(0, str(automation / "capture"))

    import capture as capture_mod

    assert hasattr(capture_mod, "find_device_index"), \
        "capture.py has no find_device_index - name resolution is missing, " \
        "so a shifted index will silently select the wrong camera again"

    controls = capture_mod.load_capture_config(
        cfg.resolve_path("paths.controls_config", ""))
    name = controls.get("device_name", "")
    index, how = capture_mod.find_device_index(name, controls["opencv_index"])

    devices = capture_mod.list_dshow_video_devices()
    if devices:
        assert "name match" in how, (
            f"'{name}' was not matched among {devices}; fell back to a "
            f"positional index, which is exactly the fragile behaviour this "
            f"guard exists to catch")
    return f"'{name}' -> index {index} ({how})"


@check("11. example scenarios parse")
def _scenarios():
    import yaml

    directory = ROOT / "scenarios"
    files = sorted(directory.glob("*.yaml"))
    assert files, "no example scenarios found"
    for path in files:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert data.get("id"), f"{path.name} has no id"
        assert data.get("success_criteria"), f"{path.name} has no success criteria"
    return f"{len(files)} scenarios valid"


# ===========================================================================
def main() -> int:
    print("\n" + "=" * 72)
    print("  SMOKE TEST - framework wiring (no hardware, no API key needed)")
    print("=" * 72 + "\n")

    for line in PASSED:
        print(f"  [PASS] {line}")
    for name, error in FAILED:
        print(f"  [FAIL] {name}")
        print(f"         {error}")

    print("\n" + "=" * 72)
    print(f"  {len(PASSED)} passed, {len(FAILED)} failed")
    print("=" * 72)

    if FAILED:
        print("\nRe-run with -v for full tracebacks.")
        return 1

    print("\nWiring is sound. Next, with the rig connected:")
    print("    python console.py health")
    return 0


if __name__ == "__main__":
    sys.exit(main())
