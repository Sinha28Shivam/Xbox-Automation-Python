"""
state.py - the shared blackboard passed between graph nodes.

DESIGN
------
LangGraph nodes are pure-ish functions: they receive the state and return a
PARTIAL dict, which LangGraph merges in. So a node never mutates state directly;
it returns only the keys it changed. That makes each agent independently
testable - you hand it a dict and inspect what comes back.

Most keys use "last write wins". Three do not, and they use reducers:

  messages        append-only  - the conversation/transcript
  agent_outputs   dict merge   - each agent writes under its own role key
  artifacts       append-only  - every file produced during the run

Append-only matters because two agents can legitimately add to the transcript,
and a plain overwrite would silently discard one agent's contribution.

WHY A TypedDict AND NOT A PYDANTIC MODEL
----------------------------------------
LangGraph merges partial dicts. A TypedDict gives editor/type-checker support
while staying an ordinary dict at runtime, so partial updates stay cheap. The
values inside are pydantic models, so the important data is still validated.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from schemas import (
    ExecutionResult,
    HealthReport,
    RequirementItem,
    RootCauseAnalysis,
    TestPlan,
    TestReport,
    ValidatedScenario,
    VerificationResult,
)


def merge_dicts(left: dict[str, Any] | None,
                right: dict[str, Any] | None) -> dict[str, Any]:
    """Reducer: shallow-merge, right wins. Used for agent_outputs."""
    return {**(left or {}), **(right or {})}


def keep_max(left: int | None, right: int | None) -> int:
    """Reducer for counters that must never go backwards.

    Step counts guard against infinite loops. If a node returned a stale lower
    value the guard would reset and the loop could run forever, so we always
    keep the highest value seen.
    """
    return max(left or 0, right or 0)


class AgenticState(TypedDict, total=False):
    """Everything the workflow knows. All keys optional; nodes fill them in."""

    # -- run identity ------------------------------------------------------
    run_id: str
    started_at: str
    scenario_input: str          # raw text or path, exactly as the user gave it
    scenario_source: str         # "file" | "text" | "requirement_file"

    # -- agent results (one key per pipeline stage) ------------------------
    requirement: RequirementItem | None
    health: HealthReport | None
    scenario: ValidatedScenario | None
    plan: TestPlan | None
    execution: ExecutionResult | None
    verification: VerificationResult | None
    rca: RootCauseAnalysis | None
    report: TestReport | None
    recovery: dict[str, Any] | None

    # -- accumulating channels (reducers) ----------------------------------
    messages: Annotated[list[dict[str, Any]], operator.add]
    agent_outputs: Annotated[dict[str, Any], merge_dicts]
    artifacts: Annotated[list[str], operator.add]
    errors: Annotated[list[str], operator.add]

    # -- control flow ------------------------------------------------------
    # Bounded so a routing bug cannot drive real hardware in a loop forever.
    step_count: Annotated[int, keep_max]
    replan_count: Annotated[int, keep_max]
    recovery_count: Annotated[int, keep_max]
    next_agent: str              # only consulted in 'supervised' route mode
    should_stop: bool
    stop_reason: str

    # -- runtime context (set once at startup, read by every node) ---------
    config: dict[str, Any]
    run_dir: str
    dry_run: bool


def initial_state(run_id: str, scenario_input: str, scenario_source: str,
                  config: dict[str, Any], run_dir: str,
                  dry_run: bool = False, started_at: str = "") -> AgenticState:
    """A fully populated starting state.

    Every key is initialised, including the accumulators. LangGraph tolerates
    missing keys, but a node that does `state["messages"] + [x]` on an absent
    key raises KeyError - so we never leave them unset.
    """
    return AgenticState(
        run_id=run_id,
        started_at=started_at,
        scenario_input=scenario_input,
        scenario_source=scenario_source,
        requirement=None,
        health=None,
        scenario=None,
        plan=None,
        execution=None,
        verification=None,
        rca=None,
        report=None,
        recovery=None,
        messages=[],
        agent_outputs={},
        artifacts=[],
        errors=[],
        step_count=0,
        replan_count=0,
        recovery_count=0,
        next_agent="",
        should_stop=False,
        stop_reason="",
        config=config,
        run_dir=run_dir,
        dry_run=dry_run,
    )


def note(role: str, text: str, **extra: Any) -> dict[str, Any]:
    """Build one transcript entry.

    The transcript is what makes a run auditable after the fact: who said what,
    in order. The RCA agent reads it, and it is embedded in the final report.
    """
    entry = {"role": role, "text": text}
    entry.update(extra)
    return entry


def state_digest(state: AgenticState) -> dict[str, Any]:
    """A small, JSON-safe summary of the state.

    Used for prompts and logging. The full state contains numpy frames and
    large nested models; passing all of that to an LLM would be expensive and
    mostly noise, so agents receive this digest instead.
    """
    health = state.get("health")
    requirement = state.get("requirement")
    scenario = state.get("scenario")
    plan = state.get("plan")
    execution = state.get("execution")
    verification = state.get("verification")

    return {
        "run_id": state.get("run_id", ""),
        "step_count": state.get("step_count", 0),
        "replan_count": state.get("replan_count", 0),
        "dry_run": state.get("dry_run", False),
        "health": None if health is None else {
            "healthy": health.healthy,
            "gimx_reachable": health.gimx_reachable,
            "capture_has_signal": health.capture_has_signal,
            "blocking_issues": health.blocking_issues,
        },
        "requirement": None if requirement is None else {
            "id": requirement.id,
            "title": requirement.title,
            "goal": requirement.goal,
            "expected_outcome": requirement.expected_outcome,
        },
        "scenario": None if scenario is None else {
            "id": scenario.id,
            "title": scenario.title,
            "goal": scenario.goal,
            "valid": scenario.valid,
            "criteria": [c.description for c in scenario.success_criteria],
        },
        "plan": None if plan is None else {
            "revision": plan.revision,
            "steps": len(plan.steps),
        },
        "execution": None if execution is None else {
            "completed": execution.completed,
            "aborted": execution.aborted,
            "abort_reason": execution.abort_reason,
            "steps_run": len(execution.steps),
            "observed_any_change": execution.observed_any_change,
        },
        "verification": None if verification is None else {
            "verdict": verification.verdict.value,
            "passed": verification.passed,
            "summary": verification.summary,
        },
        "errors": state.get("errors", [])[-5:],
    }
