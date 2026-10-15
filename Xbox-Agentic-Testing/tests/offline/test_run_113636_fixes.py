"""Regression tests for the four defects found in run-20261015-113636.

A. planner dropped `require_focus` from navigation_guidance
B. guided nav scrolled an already-open page looking for its own heading
C. dashboard guard aborted a dashboard scenario on its expected final screen
D. RCA labelled a framework-guard abort as product_defect
E. the lesson pointed at the first (recovered) error, not the abort cause
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agents.executor_agent import ExecutorAgent
from agents.planner_agent import PlannerAgent
from agents.rca_agent import RCAAgent
from core import lesson_store
from core.schemas import (ExecutionResult, FailureClass, PlannedStep,
                          RootCauseAnalysis, StepResult, SuccessCriterion,
                          TestPlan, ValidatedScenario)
from tools.navigation_tools import _is_page_heading

# ---- A ---------------------------------------------------------------------
scenario = ValidatedScenario(
    id="s", title="t", goal="g", tags=["dashboard", "guide"],
    success_criteria=[SuccessCriterion(
        description="Game Activity", check_type="vision_judgement",
        parameters={"navigation_guidance": {
            "target_text": "Game Activity", "direction": "right",
            "max_presses": 3, "interval": 0.8, "require_focus": False}})])
plan = TestPlan(scenario_id="s", steps=[PlannedStep(
    index=0, action="guided_navigate_to_text", intent="x", expected_observation="x",
    arguments={"target_text": "Game Activity", "direction": "right",
               "max_presses": 3, "interval": 0.8})])        # exactly what the LLM emitted
PlannerAgent._apply_navigation_guidance(plan, scenario)
assert plan.steps[0].arguments["require_focus"] is False, plan.steps[0].arguments

# ---- B (real OCR first lines from the run) ---------------------------------
achievements_page = "Achievements\n64,800 6\nDundecns Il\n11:08 PM"
game_activity_panel = "A\n64,800 6\n11:09 PM\nGame activity\nAchievements\nStats"
assert _is_page_heading(achievements_page, "achievements", False)
assert not _is_page_heading(game_activity_panel, "achievements", False), \
    "a list item inside Game activity must still require focus"
assert not _is_page_heading(game_activity_panel, "stats", False)

# ---- C ---------------------------------------------------------------------
assert ExecutorAgent._dashboard_is_expected(scenario)
game = ValidatedScenario(id="g", title="t", goal="g", tags=["xbox", "dungeons", "gameplay"])
assert not ExecutorAgent._dashboard_is_expected(game), "game scenarios keep the guard"
assert not ExecutorAgent._dashboard_is_expected(None)

# ---- D ---------------------------------------------------------------------
aborted = ExecutionResult(
    scenario_id="s", aborted=True,
    abort_reason="Blind tapping prevented: Console is on the Xbox Dashboard Home ('sponsored')",
    steps=[StepResult(index=0, action="press_button", dispatched=True, screen_delta=40.0)])
rca = RootCauseAnalysis(scenario_id="s", failure_class=FailureClass.PRODUCT_DEFECT,
                        primary_cause="console misbehaved", product_bug_suspected=True)
fixed = RCAAgent._apply_guardrails(
    RCAAgent.__new__(RCAAgent), rca, {"execution": aborted},
    {"gimx": {"reachable": True}, "capture": {"has_signal": True}})
assert fixed.failure_class == FailureClass.AUTOMATION_DEFECT
assert fixed.product_bug_suspected is False

# ---- E (shape of the real run: error at 6, recovered, abort at 32) ---------
steps = [StepResult(index=i, action="capture_frame") for i in range(33)]
steps[6] = StepResult(index=6, action="guided_navigate_to_text", dispatched=True,
                      error="Did not reach 'Game Activity'")
steps[31] = StepResult(index=31, action="press_button", dispatched=True,
                       arguments={"button": "b"})
steps[32] = StepResult(index=32, action="wait_for_stable_screen",
                       error="Blind tapping prevented")
real = ExecutionResult(scenario_id="s", aborted=True, steps=steps,
                       abort_reason="Blind tapping prevented")
step = lesson_store._failed_step(real.model_dump(mode="json"))
assert step["index"] == 31 and step["arguments"] == {"button": "b"}, step

print("run-20261015-113636 regression tests passed (A-E)")
