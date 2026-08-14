"""
planner_agent.py - agent 3: scenario -> ordered, verifiable steps.

Separate from the executor on purpose. Planning is reasoning; execution touches
hardware. Splitting them means a replan after a failure re-thinks without
re-running side effects, and it means the plan can be reviewed (or dry-run)
before a single button is pressed.

WHAT MAKES A GOOD PLAN HERE
---------------------------
Every step carries an `expected_observation` - what should be visible on screen
afterwards. The schema refuses steps without one. That single requirement is
what makes verification possible at all: a step with no expected outcome can be
executed but never checked, and a plan full of those produces a run that cannot
fail.

The planner is told what tools and controls exist by querying the rig at plan
time, so it can only reference things that are really there. It never sees a
hardcoded button list.

REPLANNING
----------
When the verifier sends work back, the previous plan and the failure reason are
included in the prompt. The point is a genuinely different approach - slower
timing, a different route through the menus - rather than the same plan retried
in the hope that the console changes its mind.
"""

from __future__ import annotations

from typing import Any

from base import BaseAgent
from schemas import TestPlan
from state import AgenticState, note


class PlannerAgent(BaseAgent):
    """Turns a validated scenario into executable, verifiable steps."""

    role = "planner"

    def run(self, state: AgenticState) -> dict[str, Any]:
        scenario = state.get("scenario")
        if scenario is None:
            raise ValueError(
                "Planner ran before the scenario was validated - check the "
                "edges in graph.yaml.")

        previous = state.get("plan")
        verification = state.get("verification")
        replan_count = int(state.get("replan_count", 0))
        is_replan = previous is not None and verification is not None

        prompt = self.render_prompt(
            state,
            scenario=scenario.model_dump(mode="json"),
            is_replan=is_replan,
            replan_count=replan_count,
            previous_plan=previous.model_dump(mode="json") if previous else None,
            failure_feedback=(
                verification.model_dump(mode="json") if verification else None),
            # Handing the planner the measured timings from controls.yaml stops
            # it inventing delays. game_launch_wait is documented there as a
            # placeholder rather than a measurement, and the prompt says so.
            timings=self.call_tool("get_timing").get("timings", {}),
            max_steps=int(scenario.max_steps
                          or self.context.settings.get("runtime.max_steps", 40)),
        )

        plan = self.invoke_structured(TestPlan, prompt)
        plan.scenario_id = scenario.id
        plan.revision = replan_count + 1
        if is_replan and verification is not None:
            plan.replan_reason = verification.replan_hint or verification.summary

        if not plan.steps:
            raise ValueError(
                "The planner produced no steps. The scenario may be "
                "impossible with the controls this rig exposes.")

        # Renumber defensively: models occasionally emit duplicate or
        # out-of-order indices, and later code joins step results by index.
        for i, step in enumerate(plan.steps):
            step.index = i

        self.context.artifacts.save_json(
            f"plan-r{plan.revision}.json", plan.model_dump(mode="json"))

        summary = (f"Plan revision {plan.revision}: {len(plan.steps)} steps"
                   + (f" (replan: {plan.replan_reason})" if is_replan else ""))

        return {
            "plan": plan,
            "replan_count": replan_count + (1 if is_replan else 0),
            "messages": [note(self.role, summary)],
            "agent_outputs": {self.role: {
                "ok": True,
                "steps": len(plan.steps),
                "revision": plan.revision,
                "assumptions": plan.assumptions,
            }},
        }
