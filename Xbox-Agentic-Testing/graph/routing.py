"""
routing.py - the conditional-edge functions named in graph.yaml.

Each is a pure function of the state returning a string, which graph.yaml maps
to a target node:

    - from: "verifier"
      router: "routing.route_after_verification"
      map: { passed: "reporter", failed: "rca", replan: "planner" }

WHY PURE FUNCTIONS
------------------
Routing decides whether real hardware gets driven again. Keeping these as pure
state -> string functions means the entire control flow can be unit-tested with
dictionaries - no console, no capture card, no API key. Given how expensive it
is to reproduce a hardware state, that testability matters more here than
elsewhere.

They are also deliberately conservative. Every ambiguous case routes towards
the reporter, because ending the run with an honest INCONCLUSIVE is always
better than looping against a console that is not responding.
"""

from __future__ import annotations

from typing import Any

from state import AgenticState


def _limit(state: AgenticState, key: str, fallback: int) -> int:
    """Read a runtime limit out of the config carried in the state."""
    runtime = (state.get("config") or {}).get("runtime") or {}
    try:
        return int(runtime.get(key, fallback))
    except (TypeError, ValueError):
        return fallback


def _recovery_enabled(state: AgenticState) -> bool:
    agents = (state.get("config") or {}).get("agents") or {}
    return bool((agents.get("recovery") or {}).get("enabled", False))


def _halted(state: AgenticState) -> bool:
    """Has something already decided the run cannot continue?

    Any agent may set `should_stop` - a required agent crashing, an invalid
    scenario, an exhausted budget. Every router checks this FIRST, because
    without it the graph marches on and downstream agents fail one by one on
    missing state, burying the original cause under a pile of secondary errors.
    """
    return bool(state.get("should_stop"))


def _pointless_to_retry(state: AgenticState) -> bool:
    """Would another attempt produce different evidence?

    In a dry run, no. Nothing is dispatched and no frames are captured, so a
    second plan is judged against exactly the same absence of evidence and
    reaches exactly the same INCONCLUSIVE verdict - having spent another round
    of LLM calls to get there.
    """
    return bool(state.get("dry_run"))


# ===========================================================================
# After the scenario validator / planner
# ===========================================================================
def route_after_scenario(state: AgenticState) -> str:
    """Plan only if we have a valid scenario to plan for."""
    if _halted(state):
        return "stopped"
    scenario = state.get("scenario")
    if scenario is None or not scenario.valid:
        return "stopped"
    return "plan"


def route_after_planner(state: AgenticState) -> str:
    """Execute only if a usable plan exists."""
    if _halted(state):
        return "stopped"
    plan = state.get("plan")
    if plan is None or not plan.steps:
        return "stopped"
    return "execute"


# ===========================================================================
# After health
# ===========================================================================
def route_after_health(state: AgenticState) -> str:
    """healthy -> run the test | recoverable -> try a fix | blocked -> report.

    A blocked rig goes straight to the report. It never reaches the executor,
    so it can never produce a PASS or a FAIL - only BLOCKED. That is the point:
    with a broken instrument, any statement about the console is unfounded.
    """
    health = state.get("health")
    if health is None or _halted(state):
        return "blocked"

    if health.healthy:
        return "healthy"

    if health.recoverable and _recovery_enabled(state):
        if int(state.get("recovery_count", 0)) < 1:
            return "recoverable"
    return "blocked"


# ===========================================================================
# After recovery
# ===========================================================================
def route_after_recovery(state: AgenticState) -> str:
    """Re-probe after any successful remediation; never trust it blindly."""
    recovery = state.get("recovery") or {}
    if recovery.get("recovered") and not recovery.get("requires_human"):
        return "recovered"          # loops back to health for a real re-check
    return "failed"


# ===========================================================================
# After the executor
# ===========================================================================
def route_after_executor(state: AgenticState) -> str:
    """Verify unless execution collapsed entirely.

    An abort where nothing ever changed on screen skips verification: there is
    nothing to verify, and RCA is the useful next step. An abort part-way
    through still goes to the verifier, because partial evidence can still
    disprove a criterion.
    """
    execution = state.get("execution")
    if execution is None or _halted(state):
        return "aborted"

    if execution.aborted and not execution.observed_any_change:
        return "aborted"
    return "verify"


# ===========================================================================
# After verification
# ===========================================================================
def route_after_verification(state: AgenticState) -> str:
    """passed -> report | replan -> plan again | failed -> RCA."""
    verification = state.get("verification")
    if verification is None:
        return "failed"

    if verification.passed:
        return "passed"

    # Replanning is only worth it when the fault looks like ours and the
    # console is demonstrably alive. Both conditions are required: retrying
    # against an unresponsive console just repeats the same failure.
    if verification.should_replan and not _halted(state) \
            and not _pointless_to_retry(state):
        if int(state.get("replan_count", 0)) < _limit(state, "max_replans", 3):
            execution = state.get("execution")
            if execution is None or execution.observed_any_change:
                return "replan"
    return "failed"


# ===========================================================================
# After RCA
# ===========================================================================
def route_after_rca(state: AgenticState) -> str:
    """Retry only for causes that a retry could plausibly resolve."""
    rca = state.get("rca")
    if rca is None or _halted(state) or _pointless_to_retry(state):
        return "report"

    if not rca.is_retryable:
        return "report"

    if int(state.get("replan_count", 0)) >= _limit(state, "max_replans", 3):
        return "report"

    # Rig faults and product defects are never retried here. A rig fault needs
    # a human, and a genuine console defect will reproduce every time - burning
    # attempts on either just delays the report that someone needs to read.
    if rca.failure_class.value in ("rig_fault", "product_defect"):
        return "report"
    return "retry"


# ===========================================================================
# Supervised mode
# ===========================================================================
def route_supervised(state: AgenticState) -> str:
    """Follow the supervisor's choice, with hard stops honoured first."""
    if state.get("should_stop"):
        return "reporter"
    if int(state.get("step_count", 0)) >= _limit(state, "max_steps", 40):
        return "reporter"
    return str(state.get("next_agent") or "reporter")


def should_continue(state: AgenticState) -> str:
    """Generic guard usable from any node in supervised mode."""
    return "stop" if state.get("should_stop") else "continue"
