"""Integration test: the REAL RCA, planner and verifier agents drive the lesson
store across two runs. The LLM and hardware are stubbed; everything else -
agent code, prompt templates, settings, artifact paths - is the production path.

Run 1 fails  -> RCAAgent records a lesson in artifacts/lessons/.
Run 2 plans  -> PlannerAgent loads it, the rendered prompt contains it,
                lessons-applied-r1.json is written, cached route is skipped.
Run 2 passes -> VerifierAgent resolves it.
Run 3 plans  -> no lessons in the prompt.
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.base import AgentContext
from agents.planner_agent import PlannerAgent
from agents.rca_agent import RCAAgent
from agents.verifier_agent import VerifierAgent
from core.artifacts import ArtifactStore
from core.config import Config
from core.schemas import (ExecutionResult, FailureClass, PlannedStep,
                          RootCauseAnalysis, StepResult, TestPlan,
                          ValidatedScenario, Verdict, VerificationResult)

SID = "learning-pipeline-test"
tmp = Path(tempfile.mkdtemp())
captured_prompts: list[str] = []


class _Prompts:
    """Real Jinja templates from config/prompts, recording what was rendered."""
    def __init__(self):
        import jinja2
        self.env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(ROOT / "config" / "prompts")))

    def render(self, name, ctx):
        text = self.env.get_template(name).render(**ctx)
        captured_prompts.append(text)
        return text


class _Tools:
    """Only what these three agents call; no hardware."""
    class context:
        class hardware:
            @staticmethod
            def controls_summary():
                return {}

    def describe(self, selectors=None):
        return [{"name": n, "description": "", "parameters": {}}
                for n in ("capture_frame", "guided_navigate_to_text", "press_button",
                          "check_for_text", "read_screen_text")]

    def build(self, name):
        results = {
            "check_gimx_session": {"ok": True, "reachable": True},
            "check_capture_device": {"ok": True, "has_signal": True},
            "get_timing": {"ok": True, "timings": {}},
            "read_screen_text": {"ok": True, "text": "Home"},
        }
        return lambda **_: results.get(name, {"ok": True})


def _context(run_id: str) -> AgentContext:
    settings = Config.load(ROOT / "config" / "settings.yaml")
    agents = Config.load(ROOT / "config" / "agents.yaml")
    store = ArtifactStore(tmp, run_id)
    return AgentContext(settings, agents, llm_factory=None, prompts=_Prompts(),
                        tools=_Tools(), artifacts=store)


scenario = ValidatedScenario(id=SID, title="t", goal="reach Game Activity",
                             success_criteria=[], max_steps=20)
plan = TestPlan(scenario_id=SID, revision=1, steps=[
    PlannedStep(index=0, action="capture_frame",
                expected_observation="screen captured", intent="observe")])
execution = ExecutionResult(
    scenario_id=SID, aborted=True, abort_reason="Console is on Dashboard Home",
    steps=[StepResult(index=0, action="guided_navigate_to_text",
                      arguments={"direction": "right", "max_presses": 3},
                      error="Did not reach 'Game Activity'", screen_delta=12.0,
                      ocr_text="Parties & chats")])
fail_verdict = VerificationResult(scenario_id=SID, verdict=Verdict.FAIL)

# ---- Run 1: failure -> lesson recorded by the real RCA agent -------------
ctx1 = _context("run-1")
rca_agent = RCAAgent(ctx1)
rca_agent.invoke_structured = lambda schema, prompt, images=None: RootCauseAnalysis(
    scenario_id=SID, failure_class=FailureClass.AUTOMATION_DEFECT,
    primary_cause="Right x3 from Home lands on Parties and chats.")
out = rca_agent.run({"scenario": scenario, "execution": execution,
                     "verification": fail_verdict, "plan": plan, "run_id": "run-1"})
assert "lesson L001" in out["messages"][0]["text"], out["messages"]
store = tmp / "lessons" / f"{SID}.json"
assert store.is_file()
assert (tmp / "runs" / "run-1" / "reports" / "lesson-recorded-r1.json").is_file()

# ---- Run 2: planner must load the lesson and skip any cached route ---------
routes = tmp / "routes"
routes.mkdir()
(routes / f"{SID}.route.json").write_text(plan.model_dump_json())

ctx2 = _context("run-2")
planner = PlannerAgent(ctx2)
planner.invoke_structured = lambda schema, prompt, images=None: TestPlan(
    scenario_id=SID, steps=list(plan.steps))
planner._sanitise_plan = lambda p, s=None: p
captured_prompts.clear()
out = planner.run({"scenario": scenario, "run_id": "run-2"})
assert out["lessons_applied"] == ["L001" + store.read_text().split('"L001')[1].split('"')[0]], out["lessons_applied"]
assert not out["agent_outputs"]["planner"].get("cached_route"), \
    "a cached route must not bypass open lessons"
prompt = next(p for p in captured_prompts if "TEST PLANNER" in p)
assert "LESSONS FROM EARLIER FAILED RUNS" in prompt
assert "Right x3 from Home lands on Parties and chats" in prompt
applied = json.loads((tmp / "runs" / "run-2" / "reports" / "lessons-applied-r1.json").read_text())
assert applied["count"] == 1

# ---- Enforcement: repeating the failing step without citing the lesson ----
lesson_id = out["lessons_applied"][0]
bad_step = PlannedStep(index=0, action="guided_navigate_to_text",
                       arguments={"direction": "right", "max_presses": 3},
                       expected_observation="x", intent="x")
calls = {"n": 0}

def _llm(schema, prompt, images=None):
    calls["n"] += 1
    if calls["n"] == 1:
        return TestPlan(scenario_id=SID, steps=[bad_step])
    assert "Your plan repeats that exact step" in prompt, "rejection must be fed back"
    return TestPlan(scenario_id=SID, steps=[bad_step],
                    assumptions=[f"{lesson_id}: screen now settles before each read"])

ctx2b = _context("run-2b")
planner_b = PlannerAgent(ctx2b)
planner_b.invoke_structured = _llm
planner_b._sanitise_plan = lambda p, s=None: p
planner_b.run({"scenario": scenario, "run_id": "run-2b"})
assert calls["n"] == 2, "uncited repeat must be rejected and re-asked once"

# ---- Run 2 passes: real verifier resolves the lesson ----------------------
verifier = VerifierAgent(ctx2)
verifier._emit(VerificationResult.model_construct(
    scenario_id=SID, verdict=Verdict.PASS, passed=True, criteria=[],
    evidence=[], not_proven=[], failed_steps=[], summary="ok", confidence=1.0,
    should_replan=False, replan_hint=None, stage_status={}, last_proven_stage=None),
    {"scenario": scenario, "plan": plan, "run_id": "run-2"})
assert all(l["status"] == "resolved" for l in json.loads(store.read_text())["lessons"])

# ---- Run 3: nothing open -> prompt has no lessons section ------------------
ctx3 = _context("run-3")
planner3 = PlannerAgent(ctx3)
planner3.invoke_structured = planner.invoke_structured
planner3._sanitise_plan = planner._sanitise_plan
(routes / f"{SID}.route.json").unlink(missing_ok=True)
captured_prompts.clear()
out = planner3.run({"scenario": scenario, "run_id": "run-3"})
assert out.get("lessons_applied") == []
assert not any("LESSONS FROM EARLIER FAILED RUNS" in p for p in captured_prompts)

# ---- Replan path (verifier -> planner skips RCA) still records ------------
replan_sid = SID + "-replan"
replan_scenario = scenario.model_copy(update={"id": replan_sid})
verifier._emit(VerificationResult(
    scenario_id=replan_sid, verdict=Verdict.FAIL, should_replan=True,
    replan_hint="Focus stayed on Achievements; move down before pressing A."),
    {"scenario": replan_scenario, "plan": plan, "execution": execution,
     "run_id": "run-4"})
replan_store = tmp / "lessons" / f"{replan_sid}.json"
lessons = json.loads(replan_store.read_text())["lessons"]
assert len(lessons) == 1 and "move down before pressing A" in lessons[0]["primary_cause"]

# ---- Shared lesson reaches a NEW scenario's real planner ------------------
guide_a = scenario.model_copy(update={"id": "guide-origin", "tags": ["guide", "dashboard"]})
rca5 = RCAAgent(_context("run-5"))
rca5.invoke_structured = lambda schema, prompt, images=None: RootCauseAnalysis(
    scenario_id="guide-origin", failure_class=FailureClass.AUTOMATION_DEFECT,
    primary_cause="Pressing Guide while the Guide is open closes it.")
rca5.run({"scenario": guide_a, "execution": execution, "verification": fail_verdict,
          "plan": plan, "run_id": "run-5"})

guide_new = scenario.model_copy(update={"id": "brand-new-guide-test", "tags": ["guide"]})
planner6 = PlannerAgent(_context("run-6"))
# Repeats the origin's failing step with no citation: allowed for SHARED lessons.
planner6.invoke_structured = lambda schema, prompt, images=None: TestPlan(
    scenario_id="brand-new-guide-test", steps=[bad_step])
planner6._sanitise_plan = lambda p, s=None: p
captured_prompts.clear()
out = planner6.run({"scenario": guide_new, "run_id": "run-6"})
prompt = next(p for p in captured_prompts if "TEST PLANNER" in p)
assert "Pressing Guide while the Guide is open closes it" in prompt
assert '"scope": "shared"' in prompt and "guide-origin" in prompt
assert len(out["lessons_applied"]) == 1

shutil.rmtree(tmp, ignore_errors=True)
print("learning pipeline integration test passed (real RCA, planner, verifier)")
