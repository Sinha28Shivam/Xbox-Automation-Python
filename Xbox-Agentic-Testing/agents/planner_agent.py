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

import re
from typing import Any

from base import BaseAgent
from route_store import invalidate_cached_route, load_cached_route
from schemas import PlannedStep, ScenarioStage, TestPlan, ValidatedScenario
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

        # Tier-1 Route Caching: check for verified route on initial run
        if not is_replan:
            cached_plan = load_cached_route(self.context.artifacts.run_dir, scenario.id)
            if cached_plan is not None:
                cached_plan.scenario_id = scenario.id
                cached_plan.revision = 1
                for i, step in enumerate(cached_plan.steps):
                    step.index = i
                self.context.artifacts.save_json(
                    "plan-r1.json", cached_plan.model_dump(mode="json"))
                summary = (f"Plan revision 1: {len(cached_plan.steps)} steps "
                           "(replayed verified cached route; 0ms planning latency)")
                return {
                    "plan": cached_plan,
                    "replan_count": 0,
                    "messages": [note(self.role, summary)],
                    "agent_outputs": {self.role: {
                        "ok": True,
                        "steps": len(cached_plan.steps),
                        "revision": 1,
                        "cached_route": True,
                        "assumptions": cached_plan.assumptions,
                    }},
                }
        else:
            # Invalidate failed cached route during replan
            invalidate_cached_route(self.context.artifacts.run_dir, scenario.id)

        executor_selectors = list(
            self.context.spec_for("executor").get("tools") or [])
        executor_tools = self.context.tools.describe(executor_selectors)

        prompt = self.render_prompt(
            state,
            scenario=scenario.model_dump(mode="json"),
            tools=executor_tools,
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
        if not plan.steps and scenario.stages:
            plan = self._fallback_staged_plan(scenario, plan)
        plan = self._sanitise_plan(plan)

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

    def _sanitise_plan(self, plan: TestPlan) -> TestPlan:
        """Reject impossible actions before the executor touches hardware."""
        available = {
            tool["name"] for tool in self.context.tools.describe(
                list(self.context.spec_for("executor").get("tools") or [])
            )
        }
        for step in plan.steps:
            if step.action == "verify_no_error_dialog":
                if "check_for_text" not in available:
                    raise ValueError(
                        "Planner emitted verify_no_error_dialog, but this rig "
                        "does not expose check_for_text for the safer rewrite.")
                step.action = "check_for_text"
                step.arguments = {
                    "text_patterns": [
                        "something went wrong",
                        "error",
                        "try again",
                        "not available",
                        "can't launch",
                    ],
                    "match_type": "any",
                }

            if step.action not in available:
                raise ValueError(
                    f"Planner emitted unsupported action '{step.action}'. "
                    f"Available tools: {', '.join(sorted(available))}")
        self._validate_stage_discipline(plan)
        return plan

    def _fallback_staged_plan(self, scenario: ValidatedScenario,
                              plan: TestPlan) -> TestPlan:
        """Deterministic bootstrap when the LLM returns an empty staged plan."""
        steps: list[PlannedStep] = []
        game_name = self._infer_game_name(scenario)
        target_label = self._infer_target_label(scenario)

        def add(action: str, intent: str, expected: str,
                arguments: dict[str, Any] | None = None,
                stage: ScenarioStage | None = None,
                replan_on: list[str] | None = None,
                progress_signal: str = "") -> None:
            steps.append(PlannedStep(
                index=len(steps),
                action=action,
                arguments=arguments or {},
                intent=intent,
                expected_observation=expected,
                stage=stage,
                stage_goal=self._stage_goal(scenario, stage),
                replan_on=replan_on or [],
                progress_signal=progress_signal,
            ))

        add(
            "capture_frame",
            "Capture the starting dashboard state before any action.",
            "The current starting screen is captured for later comparison.",
            {"label": "stage-preflight-baseline"},
            ScenarioStage.PREFLIGHT,
            progress_signal="baseline_captured",
        )

        if game_name:
            add(
                "launch_game",
                f"Identify and launch {game_name} only after its tile is visually matched.",
                f"The focused tile is proven to be {game_name}, then the game launch is dispatched.",
                {"game_name": game_name, "max_tiles": 2, "launch_wait": 8.0},
                ScenarioStage.GAME_DISCOVERY,
                ["game_not_found", "wrong_game_detected"],
                "game_discovered",
            )
            add(
                "wait_for_stable_screen",
                f"Wait for {game_name} to settle after launch.",
                f"A stable {game_name} startup, menu, or gameplay screen is visible.",
                {"label": "stage-game-launch-stable"},
                ScenarioStage.GAME_LAUNCH,
                ["launch_unproven"],
                "game_launch_screen",
            )

        add(
            "press_button",
            "Dismiss the start screen prompt and advance to the game menu.",
            "The start screen prompt is dismissed and the game enters the menu.",
            {"button": "a"},
            ScenarioStage.MENU_DETECTION,
            progress_signal="menu_entered",
        )
        add(
            "wait_for_stable_screen",
            "Wait for the game menu to settle.",
            "The game menu is visible and stable.",
            {"label": "stage-menu-settle"},
            ScenarioStage.MENU_DETECTION,
            progress_signal="menu_settled",
        )
        add(
            "capture_frame",
            "Observe the game menu state.",
            "The current menu screen is captured for verification.",
            {"label": "stage-menu-detection"},
            ScenarioStage.LEVEL_NAVIGATION,
            progress_signal="menu_observed",
        )
        add(
            "press_button",
            f"Select {target_label} to launch the level.",
            f"The selection is confirmed on {target_label} and the screen transitions into gameplay.",
            {"button": "a"},
            ScenarioStage.LEVEL_LAUNCH,
            progress_signal="level_selected",
        )
        add(
            "wait_for_stable_screen",
            f"Wait for {target_label} gameplay to become stable.",
            f"An interactive {target_label} gameplay screen is visible.",
            {"label": "stage-level-launch-stable"},
            ScenarioStage.LEVEL_LAUNCH,
            progress_signal="interactive_gameplay",
        )
        add(
            "capture_frame",
            "Observe the gameplay state before taking the first closed-loop action.",
            "The gameplay screen is captured and ready for the next observe-decide-act cycle.",
            {"label": "stage-play-loop-observe"},
            ScenarioStage.CLOSED_LOOP_PLAY,
            progress_signal="play_loop_observation",
        )
        add(
            "move_stick",
            "Move Max forward in Anotherland to prove interactive gameplay control.",
            "Max moves forward and the gameplay scene updates visibly.",
            {"stick": "left_stick", "direction": "right", "duration": 1.5, "strength": 1.0},
            ScenarioStage.CLOSED_LOOP_PLAY,
            progress_signal="level_progress",
        )

        plan.steps = steps
        plan.assumptions = list(dict.fromkeys([
            *plan.assumptions,
            "The console starts on the Xbox dashboard.",
            "The requested game is available on tile 1 or tile 2.",
            "The visible menu route to Anotherland can be determined from on-screen evidence.",
        ]))
        plan.rationale = (
            plan.rationale or
            "Fallback staged bootstrap plan used because the LLM planner "
            "returned zero steps. This plan is intentionally conservative and "
            "keeps all confirmations evidence-first."
        )
        return plan

    @staticmethod
    def _stage_goal(scenario: ValidatedScenario,
                    stage: ScenarioStage | None) -> str:
        if stage is None:
            return ""
        for item in scenario.stages:
            if item.id == stage:
                return item.objective
        return ""

    @staticmethod
    def _infer_game_name(scenario: ValidatedScenario) -> str:
        title = str(scenario.title or "")
        match = re.match(r"(.+?)\s+-", title)
        if match:
            return match.group(1).strip()
        goal = str(scenario.goal or "")
        found = re.search(r"Prove that (.+?) can be discovered", goal)
        return found.group(1).strip() if found else ""

    @staticmethod
    def _infer_target_label(scenario: ValidatedScenario) -> str:
        for criterion in scenario.success_criteria:
            text = " ".join([
                str(criterion.description or ""),
                str(criterion.parameters.get("expected_visual", "")),
                str(criterion.parameters.get("expected_screen", "")),
            ])
            match = re.search(r"(Chapter\s+\d+:\s*[A-Za-z0-9]+)", text)
            if match:
                return match.group(1).strip()
        return "Chapter 1: Anotherland"

    @staticmethod
    def _validate_stage_discipline(plan: TestPlan) -> None:
        """Reject plans that skip proof when staged navigation needs it."""
        last_focus_proof: dict[str, int] = {}
        for step in plan.steps:
            if step.stage is not None:
                if step.action == "detect_focus_highlight":
                    last_focus_proof[step.stage.value] = step.index
                elif (
                    step.action == "press_button"
                    and str(step.arguments.get("button", "")).lower() in {"a", "cross"}
                    and step.stage.value == "level_navigation"
                ):
                    prior = last_focus_proof.get(step.stage.value)
                    if prior is None or prior != step.index - 1:
                        raise ValueError(
                            "Planner emitted a confirm action in level_navigation "
                            "without an immediately preceding focus-proof step.")
