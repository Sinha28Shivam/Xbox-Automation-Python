"""Lessons enforced DURING execution (core/runtime_lessons.py + executor wiring).

Screen texts are the real OCR from run-20261014-162340 / run-20261015-113636.
The executor test drives the REAL ExecutorAgent.run loop with fake tools, so it
proves the skip actually prevents the Guide press from being dispatched.
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import runtime_lessons as rl

L001 = {"id": "L001", "what_went_wrong":
        "Replan pressed Guide while the Guide was already open on the Achievements "
        "page, which closed it and left the console on Dashboard Home.",
        "failed_step": {"action": "press_button", "arguments": {"button": "right", "times": 1}}}
L_OTHER = {"id": "L009", "what_went_wrong": "Stats focus was missed.",
           "failed_step": {"action": "press_button", "arguments": {"button": "a"}}}

GUIDE_OPEN = "A\n64,800 6\n11:09 PM\nGame activity\nHold for power options\nAchievements"
DASHBOARD = "Game on\nExclusive deals\nIn-game benefits just for you\nSPONSORED"
GUIDE = {"button": "guide"}

# 1. Rule activation comes from lesson CONTENT.
assert rl.active_rules([L001]) == {"guide_already_open": ["L001"]}
assert rl.active_rules([L_OTHER]) == {}

# 2. Guide open -> skip the Guide press.
d = rl.decide("press_button", GUIDE, GUIDE_OPEN, rl.active_rules([L001]), [L001])
assert d["action"] == "skip" and d["rule"] == "guide_already_open", d

# 3. Dashboard (Guide closed) -> proceed.
d = rl.decide("press_button", GUIDE, DASHBOARD, rl.active_rules([L001]), [L001])
assert d["action"] == "proceed"

# 4. Without the lesson the rule does not exist: no behaviour change.
assert rl.decide("press_button", GUIDE, GUIDE_OPEN, {}, []) is None

# 5. Exact repeat of a failing step -> proceed but audited.
d = rl.decide("press_button", {"button": "a"}, DASHBOARD, {}, [L_OTHER])
assert d["action"] == "proceed" and d["rule"] == "repeat_of_failing_step"

# 6. Unrelated steps cost nothing (no OCR call).
assert not rl.needs_screen("press_button", {"button": "b"}, rl.active_rules([L001]), [L001])
assert rl.needs_screen("press_button", GUIDE, rl.active_rules([L001]), [L001])

# ---- 7. Real executor loop -------------------------------------------------
from agents.executor_agent import ExecutorAgent
from core.artifacts import ArtifactStore
from core.config import Config
from core.schemas import PlannedStep, TestPlan, ValidatedScenario

ROOT = Path(__file__).resolve().parents[1]
store = ArtifactStore(tempfile.mkdtemp(), "run-x")
(store.reports_dir / "lessons-applied-r1.json").write_text(
    json.dumps({"lessons": [L001]}), encoding="utf-8")

dispatched = []
screen = {"text": GUIDE_OPEN}


class _Ctx:
    settings = Config.load(ROOT / "config" / "settings.yaml")
    artifacts = store

    def spec_for(self, role):
        return {"tools": []}


agent = ExecutorAgent.__new__(ExecutorAgent)
agent.context = _Ctx()
agent.role = "executor"


def _call_tool(name, **kw):
    if name == "capture_frame":
        return {"ok": True, "frame_path": "x.png"}
    if name == "read_screen_text":
        return {"ok": True, "text": screen["text"]}
    return {"ok": True}


def _execute_step(step, dry_run):
    from core.schemas import StepResult
    dispatched.append((step.action, step.arguments))
    return StepResult(index=step.index, action=step.action, dispatched=True,
                      screen_delta=10.0)


agent.call_tool = _call_tool
agent._execute_step = _execute_step
agent._action_log = lambda: []

plan = TestPlan(scenario_id="s", revision=1, steps=[
    PlannedStep(index=0, action="press_button", arguments=GUIDE,
                intent="open Guide", expected_observation="Guide open"),
    PlannedStep(index=1, action="press_button", arguments={"button": "right"},
                intent="move", expected_observation="moved"),
])
scenario = ValidatedScenario(id="s", title="t", goal="g", tags=["guide"])
out = agent.run({"plan": plan, "scenario": scenario, "dry_run": False})

assert ("press_button", GUIDE) not in dispatched, "Guide must NOT be pressed when open"
assert ("press_button", {"button": "right"}) in dispatched, "rest of plan still runs"
steps = out["execution"].steps
assert steps[0].dispatched is False and "Skipped by runtime lesson L001" in steps[0].observation
assert steps[0].error is None, "a correct skip is not a failure"
log = json.loads((store.reports_dir / "runtime-lessons-r1.json").read_text())
assert log["decisions"][0]["action"] == "skip" and log["decisions"][0]["step_index"] == 0

# 8. Same plan, Guide closed -> Guide IS pressed.
dispatched.clear()
screen["text"] = DASHBOARD
agent.run({"plan": plan, "scenario": scenario, "dry_run": False})
assert ("press_button", GUIDE) in dispatched

print("runtime lesson tests passed (8 cases, incl. real executor loop)")
