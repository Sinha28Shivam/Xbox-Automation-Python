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


# Stages where a menu item is SELECTED. In every one of these, pressing A
# without having just proven what is focused risks confirming the wrong item -
# the exact failure that blocked an earlier Max run at chapter select. Any
# stage added here inherits the "prove focus, then confirm" discipline.
def _is_gameplay_action(step: Any) -> bool:
    """Is this A-press a gameplay JUMP rather than a menu CONFIRM?

    The focus-proof rule exists to stop a menu item being selected blind. Once
    a stage has reached actual gameplay, `a` means jump - there is no menu item
    to prove, and demanding a focus highlight would make a legitimate jump step
    impossible. Distinguished by the step's own declared intent/progress
    signal rather than by guessing from the stage name.
    """
    signal = str(getattr(step, "progress_signal", "") or "").lower()
    if signal in {"jump_response", "progress_signal", "level_progress",
                  "player_movement"}:
        return True
    text = " ".join([
        str(getattr(step, "intent", "") or ""),
        str(getattr(step, "expected_observation", "") or ""),
    ]).lower()
    return "jump" in text


# Stages that can only happen once the game is already running and playable.
# A scenario declaring nothing outside this set is a "resume from live
# gameplay" scenario: no discovery, no launch, no menu walking.
_IN_GAMEPLAY_ONLY_STAGES = frozenset({
    "closed_loop_play",
    "pause_checkpoint",
})


_FOCUS_PROOF_STAGES = frozenset({
    "level_navigation",
    "pause_checkpoint",
    "main_menu_return",
    "level_select_replay",
    "achievements_review",
})


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
            declared_stages = {s.id.value for s in (scenario.stages or [])}
            cached_stages = {s.stage.value for s in (cached_plan.steps if cached_plan else []) if s.stage is not None}
            if cached_plan is not None and (not declared_stages or declared_stages.issubset(cached_stages)):
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

        # Coverage/discipline failures are corrigible: the model produced a
        # structurally valid plan that simply stops too early. Retrying the
        # bare prompt is useless - it makes the same choice again - so the
        # specific complaint is appended and the model asked to try again.
        # Without this a 12-stage journey dies at exit code 4 instead of being
        # fixed by the one piece of information that would fix it.
        plan = None
        attempt_prompt = prompt
        last_error = None
        for _attempt in range(3):
            candidate = self.invoke_structured(TestPlan, attempt_prompt)
            candidate.scenario_id = scenario.id
            candidate.revision = replan_count + 1
            if is_replan and verification is not None:
                candidate.replan_reason = (
                    verification.replan_hint or verification.summary)
            if not candidate.steps and scenario.stages:
                # Loud on purpose. A silent fallback here once masked an
                # output-token truncation as a "planning failure" and cost a
                # full hardware run to diagnose.
                print(f"  [planner] WARNING: the LLM returned ZERO steps "
                      f"(attempt {_attempt + 1}). This usually means the "
                      f"response was truncated by max_tokens. Falling back to "
                      f"the deterministic staged bootstrap, which covers only "
                      f"the core stages.")
                candidate = self._fallback_staged_plan(scenario, candidate)
            try:
                plan = self._sanitise_plan(candidate, scenario)
                break
            except ValueError as exc:
                last_error = exc
                attempt_prompt = (
                    f"{prompt}\n\n# Your previous attempt was REJECTED\n\n"
                    f"{exc}\n\nProduce a corrected TestPlan that fixes exactly "
                    f"this. Keep every step you already had and ADD the "
                    f"missing ones - do not shorten the plan.")

        if plan is None:
            # Prefer the deterministic staged plan over failing the whole run:
            # a conservative plan covering the declared stages is far more
            # useful than exit code 4 with zero steps.
            if scenario.stages:
                plan = self._fallback_staged_plan(
                    scenario,
                    TestPlan(scenario_id=scenario.id,
                             revision=replan_count + 1))
                # NOTE: deliberately WITHOUT `scenario`, so the coverage guard
                # is skipped. The bootstrap only knows the core stages; running
                # it and reporting honestly which stages went untested beats
                # exit code 4 with no steps at all. The verifier will still
                # mark the unreached stages unproven.
                plan = self._sanitise_plan(plan)
                print(f"  [planner] WARNING: falling back to the deterministic "
                      f"staged bootstrap after 3 rejected attempts. It covers "
                      f"only the core stages, so later journey stages will be "
                      f"reported as untested. Last rejection: {last_error}")
            else:
                raise ValueError(
                    f"The planner could not produce an acceptable plan "
                    f"after 3 attempts: {last_error}")


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
        print(f"  [planner] Generated {len(plan.steps)} steps across {len(scenario.stages or [])} stages (Revision {plan.revision}).", flush=True)

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

    def _sanitise_plan(self, plan: TestPlan,
                       scenario: ValidatedScenario | None = None) -> TestPlan:
        """Reject impossible actions before the executor touches hardware."""
        available = {
            tool["name"] for tool in self.context.tools.describe(
                list(self.context.spec_for("executor").get("tools") or [])
            )
        }
        plan.steps = [s for s in plan.steps if s.action != "check_capture_device"]
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
        if scenario is not None:
            self._validate_stage_coverage(plan, scenario)
        return plan

    def _fallback_in_gameplay_plan(self, scenario: ValidatedScenario,
                                   plan: TestPlan) -> TestPlan:
        """Bootstrap for a scenario that starts from LIVE gameplay.

        Deliberately tiny. The interesting decisions - where to draw, when to
        jump, when to destroy a drawing, when to keep advancing - are made
        inside `vision_guided_gameplay`, one decision per captured frame. A
        long pre-baked step list would be the opposite of vision-guided: it
        would commit to actions chosen before anything had been looked at.

        So the plan is: prove we really are in gameplay, hand over to the
        vision loop, then prove the screen moved.
        """
        steps: list[PlannedStep] = []
        goal = (scenario.goal or "").strip() or (
            "Advance through the level under vision guidance until a "
            "checkpoint is reached.")

        def add(action: str, intent: str, expected: str,
                arguments: dict[str, Any] | None = None,
                progress_signal: str = "",
                timeout_seconds: float | None = None) -> None:
            steps.append(PlannedStep(
                index=len(steps),
                action=action,
                arguments=arguments or {},
                intent=intent,
                expected_observation=expected,
                stage=ScenarioStage.CLOSED_LOOP_PLAY,
                stage_goal=self._stage_goal(scenario, ScenarioStage.CLOSED_LOOP_PLAY),
                progress_signal=progress_signal,
                timeout_seconds=timeout_seconds,
            ))

        add(
            "capture_frame",
            "Capture the live gameplay screen before touching anything, so "
            "every later frame has a baseline to be compared against.",
            "An interactive Max: The Curse of Brotherhood gameplay frame is "
            "captured - not the dashboard, store, or a menu.",
            {"label": "gameplay-baseline"},
            progress_signal="gameplay_screen",
        )
        add(
            "read_screen_text",
            "Read any text on the live frame to confirm this is gameplay and "
            "not a store/account/dashboard screen.",
            "Either no menu text is found (plain gameplay) or the text is "
            "consistent with being inside the game.",
            {},
        )
        add(
            "vision_guided_gameplay",
            "Hand control to the vision loop: look at each frame, decide where "
            "and how to draw with the Magic Marker, when to jump, when to "
            "destroy a drawing, and how to advance - and keep playing until a "
            "checkpoint is seen or the run is stopped.",
            "Successive captured frames show Max advancing through the level, "
            "with marker strokes drawn at visible nodes, and the loop reports "
            "the observed reason it stopped.",
            {
                "goal": goal,
                "max_cycles": 40,
                "cycle_delay": 0.4,
                "stop_on_checkpoint": True,
            },
            progress_signal="progress_signal",
            # NOT the scenario timeout. The executor spends step.timeout_seconds
            # inside wait_for_stable_screen AFTER the tool returns, and a live
            # gameplay screen never becomes stable - a large value here would
            # block for the whole budget. The loop bounds its own duration via
            # max_cycles, so this only needs to be long enough to grab a
            # settled-ish closing frame.
            timeout_seconds=6.0,
        )
        add(
            "capture_frame",
            "Capture the final gameplay state the loop left the game in.",
            "A final gameplay frame is archived showing where the session "
            "ended - ideally a checkpoint or a further point in the level.",
            {"label": "gameplay-final"},
            progress_signal="progress_signal",
        )

        plan.steps = steps
        plan.assumptions = list(dict.fromkeys([
            *plan.assumptions,
            "Max: The Curse of Brotherhood is already running and in "
            "interactive gameplay when the run starts.",
            "The configured LLM provider supports vision, so frames can "
            "actually be looked at.",
            "A checkpoint is close enough to reach within the cycle budget; "
            "otherwise the run ends on the budget and says so.",
        ]))
        plan.rationale = (
            plan.rationale or
            "In-gameplay bootstrap: prove we are in live gameplay, then hand "
            "over to the frame-by-frame vision loop rather than pre-baking "
            "gameplay actions that were chosen before anything was seen.")
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

        # A scenario that declares ONLY in-gameplay stages is starting from
        # live gameplay, not from the dashboard. Emitting the launch/menu
        # bootstrap for it would walk the dashboard while the game is already
        # running - blind input at best, quitting the level at worst.
        declared = {s.id.value for s in (scenario.stages or [])}
        if declared and declared.issubset(_IN_GAMEPLAY_ONLY_STAGES):
            return self._fallback_in_gameplay_plan(scenario, plan)

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
            "Move Max forward (stick right) in gameplay to prove forward movement.",
            "Max moves forward and the gameplay scene updates visibly.",
            {"stick": "left_stick", "direction": "right", "duration": 1.2, "strength": 1.0},
            ScenarioStage.CLOSED_LOOP_PLAY,
            progress_signal="level_progress",
        )
        add(
            "move_stick",
            "Move Max backward (stick left) in gameplay to prove backward movement.",
            "Max moves backward and the gameplay scene updates visibly.",
            {"stick": "left_stick", "direction": "left", "duration": 1.2, "strength": 1.0},
            ScenarioStage.CLOSED_LOOP_PLAY,
            progress_signal="player_movement",
        )
        add(
            "press_button",
            "Jump in gameplay with A button to prove jump responsiveness.",
            "Max jumps and returns to surface with visible screen delta.",
            {"button": "a"},
            ScenarioStage.CLOSED_LOOP_PLAY,
            progress_signal="jump_response",
        )
        add(
            "draw_magic_marker",
            "Hold RT and move left stick to draw with Magic Marker holding A.",
            "Magic Marker opens via RT, draws with A + Left Stick, and returns to gameplay.",
            {"direction": "up", "duration": 1.5, "stick": "left_stick"},
            ScenarioStage.CLOSED_LOOP_PLAY,
            progress_signal="marker_interaction",
        )

        declared_stage_ids = {s.id.value for s in (scenario.stages or [])}

        if "pause_checkpoint" in declared_stage_ids:
            add(
                "press_button",
                "Open the in-game pause screen.",
                "The pause menu overlay becomes visible.",
                {"button": "start"},
                ScenarioStage.PAUSE_CHECKPOINT,
                progress_signal="pause_screen_visible",
            )
            add(
                "wait_for_stable_screen",
                "Wait for pause screen options to settle.",
                "Pause menu with Last Checkpoint option is stable.",
                {"label": "stage-pause-checkpoint-stable"},
                ScenarioStage.PAUSE_CHECKPOINT,
            )
            add(
                "detect_focus_highlight",
                "Prove Last Checkpoint is focused before selecting it.",
                "Last Checkpoint is proven to be highlighted.",
                {},
                ScenarioStage.PAUSE_CHECKPOINT,
            )
            add(
                "press_button",
                "Select Last Checkpoint.",
                "Checkpoint reload initiates.",
                {"button": "a"},
                ScenarioStage.PAUSE_CHECKPOINT,
                progress_signal="checkpoint_selected",
            )
            add(
                "wait_for_stable_screen",
                "Wait for the game to reload at the last checkpoint.",
                "Interactive gameplay is restored at the checkpoint.",
                {"label": "stage-checkpoint-restored"},
                ScenarioStage.PAUSE_CHECKPOINT,
                progress_signal="checkpoint_gameplay_resumed",
            )
            add(
                "move_stick",
                "Verify controller input responds after checkpoint reload.",
                "Max moves right and screen updates.",
                {"stick": "left_stick", "direction": "right", "duration": 1.0, "strength": 1.0},
                ScenarioStage.PAUSE_CHECKPOINT,
                progress_signal="progress_signal",
            )

        if "main_menu_return" in declared_stage_ids:
            add(
                "press_button",
                "Open the in-game pause screen to return to Main Menu.",
                "The pause menu overlay is visible.",
                {"button": "start"},
                ScenarioStage.MAIN_MENU_RETURN,
                progress_signal="pause_screen_visible",
            )
            add(
                "wait_for_stable_screen",
                "Wait for pause menu to stabilize.",
                "Pause menu options are visible.",
                {"label": "stage-pause-main-menu-stable"},
                ScenarioStage.MAIN_MENU_RETURN,
            )
            add(
                "detect_focus_highlight",
                "Prove Main Menu option is focused.",
                "Main Menu is highlighted on screen.",
                {},
                ScenarioStage.MAIN_MENU_RETURN,
            )
            add(
                "press_button",
                "Confirm selection of Main Menu.",
                "Screen transitions to Max main menu.",
                {"button": "a"},
                ScenarioStage.MAIN_MENU_RETURN,
                progress_signal="main_menu_selected",
            )
            add(
                "wait_for_stable_screen",
                "Wait for Max Main Menu screen to settle.",
                "Main Menu with Select Level option is visible.",
                {"label": "stage-main-menu-settled"},
                ScenarioStage.MAIN_MENU_RETURN,
                progress_signal="main_menu_visible",
            )

        if "level_select_replay" in declared_stage_ids:
            add(
                "detect_focus_highlight",
                "Prove Select Level is focused on the main menu.",
                "Select Level is highlighted.",
                {},
                ScenarioStage.LEVEL_SELECT_REPLAY,
            )
            add(
                "press_button",
                "Open Select Level.",
                "Level selection screen opens.",
                {"button": "a"},
                ScenarioStage.LEVEL_SELECT_REPLAY,
                progress_signal="level_select_screen_visible",
            )
            add(
                "wait_for_stable_screen",
                "Wait for chapter/level grid to appear.",
                "Chapter selection is visible.",
                {"label": "stage-level-grid-stable"},
                ScenarioStage.LEVEL_SELECT_REPLAY,
            )
            add(
                "select_level",
                "Dynamically search for target level (Sea of Sand) via OCR across chapters and select it.",
                "Level is matched via OCR, selected with A, and level loading initiates.",
                {"target_level": "Sea of Sand", "chapter": "Chapter 1", "max_attempts": 8},
                ScenarioStage.LEVEL_SELECT_REPLAY,
                progress_signal="level_launched",
            )
            add(
                "wait_for_stable_screen",
                "Wait for level gameplay to load.",
                "Interactive level gameplay is visible.",
                {"label": "stage-sea-of-sand-stable"},
                ScenarioStage.LEVEL_SELECT_REPLAY,
                progress_signal="interactive_gameplay",
            )
            add(
                "move_stick",
                "Move forward in reloaded level to verify controls.",
                "Max moves forward with visible screen delta.",
                {"stick": "left_stick", "direction": "right", "duration": 1.2, "strength": 1.0},
                ScenarioStage.LEVEL_SELECT_REPLAY,
                progress_signal="progress_signal",
            )
            add(
                "move_stick",
                "Move backward in reloaded level to verify controls.",
                "Max moves backward with visible screen delta.",
                {"stick": "left_stick", "direction": "left", "duration": 1.2, "strength": 1.0},
                ScenarioStage.LEVEL_SELECT_REPLAY,
                progress_signal="player_movement",
            )
            add(
                "press_button",
                "Jump in reloaded level with A.",
                "Max jumps with visible response.",
                {"button": "a"},
                ScenarioStage.LEVEL_SELECT_REPLAY,
                progress_signal="jump_response",
            )
            add(
                "draw_magic_marker",
                "Hold RT and move left stick to draw with Magic Marker holding A in reloaded level.",
                "Magic Marker opens via RT, draws with A + Left Stick, and returns to gameplay.",
                {"direction": "up", "duration": 1.5, "stick": "left_stick"},
                ScenarioStage.LEVEL_SELECT_REPLAY,
                progress_signal="marker_interaction",
            )

        if "achievements_review" in declared_stage_ids:
            add(
                "press_button",
                "Open menu to access Achievements.",
                "Menu is visible.",
                {"button": "start"},
                ScenarioStage.ACHIEVEMENTS_REVIEW,
                progress_signal="pause_screen_visible",
            )
            add(
                "wait_for_stable_screen",
                "Wait for menu options to settle.",
                "Menu options are visible.",
                {"label": "stage-achievements-menu-stable"},
                ScenarioStage.ACHIEVEMENTS_REVIEW,
            )
            add(
                "detect_focus_highlight",
                "Prove Achievements option is focused.",
                "Achievements option is highlighted.",
                {},
                ScenarioStage.ACHIEVEMENTS_REVIEW,
            )
            add(
                "press_button",
                "Open Achievements screen.",
                "Achievements screen is displayed.",
                {"button": "a"},
                ScenarioStage.ACHIEVEMENTS_REVIEW,
                progress_signal="achievements_screen_visible",
            )
            add(
                "wait_for_stable_screen",
                "Wait for achievements list to render.",
                "Achievements entries are visible.",
                {"label": "stage-achievements-list"},
                ScenarioStage.ACHIEVEMENTS_REVIEW,
            )
            add(
                "read_screen_text",
                "Read achievement entries on screen.",
                "Achievement titles and descriptions are extracted via OCR.",
                {},
                ScenarioStage.ACHIEVEMENTS_REVIEW,
                progress_signal="achievement_entries_readable",
            )
            add(
                "press_button",
                "Press B to back out of Achievements.",
                "Returns back to menu screen.",
                {"button": "b"},
                ScenarioStage.ACHIEVEMENTS_REVIEW,
                progress_signal="menu_screen_restored",
            )
            add(
                "wait_for_stable_screen",
                "Wait for menu to restore.",
                "Menu screen is visible again.",
                {"label": "stage-menu-restored"},
                ScenarioStage.ACHIEVEMENTS_REVIEW,
            )

        if "exit_to_dashboard" in declared_stage_ids:
            add(
                "press_button",
                "Press Xbox Guide button to open system guide overlay.",
                "Xbox guide overlay is visible.",
                {"button": "guide"},
                ScenarioStage.EXIT_TO_DASHBOARD,
                progress_signal="guide_visible",
            )
            add(
                "wait_for_stable_screen",
                "Wait for Xbox guide overlay to settle.",
                "Guide overlay is visible.",
                {"label": "stage-guide-settled"},
                ScenarioStage.EXIT_TO_DASHBOARD,
            )
            add(
                "press_button",
                "Scroll down to Max: The Curse of Brotherhood in guide.",
                "Highlight moves down to the running Max game tile in guide.",
                {"button": "down"},
                ScenarioStage.EXIT_TO_DASHBOARD,
            )
            add(
                "wait_for_stable_screen",
                "Wait for highlight to settle on Max tile in guide.",
                "Max game item is selected in guide.",
                {"label": "stage-guide-game-settled"},
                ScenarioStage.EXIT_TO_DASHBOARD,
            )
            add(
                "press_button",
                "Press Menu button on Max entry to open contextual options.",
                "Contextual pop-up menu with Quit option appears.",
                {"button": "start"},
                ScenarioStage.EXIT_TO_DASHBOARD,
                progress_signal="menu_screen_restored",
            )
            add(
                "wait_for_stable_screen",
                "Wait for context menu to appear.",
                "Context menu options are visible.",
                {"label": "stage-context-menu-settled"},
                ScenarioStage.EXIT_TO_DASHBOARD,
            )
            add(
                "press_button",
                "Navigate down to Quit option in popup menu.",
                "Quit option is highlighted.",
                {"button": "down"},
                ScenarioStage.EXIT_TO_DASHBOARD,
            )
            add(
                "press_button",
                "Confirm Quit with A to terminate game and exit to dashboard.",
                "Game terminates and Xbox dashboard is loaded.",
                {"button": "a"},
                ScenarioStage.EXIT_TO_DASHBOARD,
                progress_signal="dashboard_exit",
            )
            add(
                "wait_for_stable_screen",
                "Wait for Xbox dashboard home screen to load.",
                "Xbox dashboard home screen is visible.",
                {"label": "stage-dashboard-stable"},
                ScenarioStage.EXIT_TO_DASHBOARD,
                progress_signal="dashboard_visible",
            )
            add(
                "capture_frame",
                "Capture final dashboard state.",
                "Final dashboard screenshot is archived.",
                {"label": "stage-dashboard-final"},
                ScenarioStage.EXIT_TO_DASHBOARD,
                progress_signal="screen_change",
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
        for s in (scenario.stages or []):
            if s.id == stage:
                return s.objective
        return ""

    @staticmethod
    def _infer_game_name(scenario: ValidatedScenario) -> str:
        text = f"{scenario.title} {scenario.description} {scenario.goal}"
        for candidate in ("Max: The Curse of Brotherhood", "Minecraft", "Halo"):
            if candidate.lower() in text.lower():
                return candidate
        found = re.search(r"launch\s+([A-Za-z0-9:\s]+?)\s+(?:from|on)", text, re.IGNORECASE)
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
    def _validate_stage_coverage(plan: TestPlan,
                                 scenario: ValidatedScenario) -> None:
        """Reject a plan that abandons declared stages.

        A journey scenario declares every stage it intends to prove. A plan
        that stops early is not a partial success - it produces a run where
        most criteria are never attempted, which the verifier can only call
        INCONCLUSIVE. Catching it here costs one replan instead of a full
        multi-minute hardware run that was never going to answer the question.
        """
        declared = [s.id.value for s in (scenario.stages or [])]
        if not declared:
            return

        planned = {step.stage.value for step in plan.steps
                   if step.stage is not None}
        missing = [name for name in declared if name not in planned]
        if missing:
            raise ValueError(
                f"Plan covers only {len(planned)} of {len(declared)} declared "
                f"stages. Missing steps for: {', '.join(missing)}. Every "
                f"declared stage needs at least one step in THIS plan - a plan "
                f"that stops early leaves most success criteria untested.")

    @staticmethod
    def _validate_stage_discipline(plan: TestPlan) -> None:
        """Reject plans that skip proof when staged navigation needs it."""
        last_focus_proof: dict[str, int] = {}
        for step in plan.steps:
            if step.stage is not None:
                if step.action in {"detect_focus_highlight", "check_for_text", "read_screen_text"}:
                    last_focus_proof[step.stage.value] = step.index
                elif (
                    step.action == "press_button"
                    and str(step.arguments.get("button", "")).lower() in {"a", "cross"}
                    and step.stage.value in _FOCUS_PROOF_STAGES
                    and not _is_gameplay_action(step)
                ):
                    prior = last_focus_proof.get(step.stage.value)
                    if prior is None or prior != step.index - 1:
                        raise ValueError(
                            f"Planner emitted a confirm action in {step.stage.value} "
                            "without an immediately preceding focus-proof step.")
