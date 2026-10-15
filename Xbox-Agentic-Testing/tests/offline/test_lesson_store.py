"""Offline tests for failure learning (core/lesson_store.py) and its wiring.

Covers the full lifecycle against a temporary artifacts dir:
fail -> lesson recorded -> next plan sees it -> repeat bumps count ->
PASS resolves -> next plan sees nothing. Also checks the planner prompt
actually contains the lesson and that rig faults never steer the planner.
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import jinja2

from core import lesson_store as ls
from core.schemas import (CriterionResult, ExecutionResult, FailureClass,
                          RootCauseAnalysis, StepResult, Verdict,
                          VerificationResult)

root = Path(tempfile.mkdtemp())
run_dir = root / "runs" / "run-A"
(run_dir / "reports").mkdir(parents=True)
SID = "entry-point-game-activity"

# A guided-navigation trace like the real run-20261014-162340 one.
(run_dir / "reports" / "guided-navigation-01-game-activity.json").write_text(json.dumps({
    "target_text": "Game Activity", "direction": "right", "found": False,
    "text_seen": False, "attempts": [
        {"visible_lines": ["3:55 AM", "Home"]},
        {"visible_lines": ["People", "Friend requests"]},
        {"visible_lines": ["Parties & chats", "Start a party"]},
    ]}))

execution = ExecutionResult(
    scenario_id=SID, aborted=True,
    abort_reason="Blind tapping prevented: Console is on the Xbox Dashboard Home",
    steps=[StepResult(index=0, action="capture_frame"),
           StepResult(index=6, action="guided_navigate_to_text",
                      arguments={"target_text": "Game Activity", "direction": "right",
                                 "max_presses": 3},
                      error="Did not reach 'Game Activity'", ocr_text="Parties & chats")])
verification = VerificationResult(
    scenario_id=SID, verdict=Verdict.FAIL,
    criteria=[CriterionResult(criterion="Game Activity reached", met=False)],
    not_proven=["Game Activity section reached"])
rca = RootCauseAnalysis(
    scenario_id=SID, failure_class=FailureClass.AUTOMATION_DEFECT,
    primary_cause="Right x3 from Home lands on Parties & chats, not Game Activity.",
    recommendations=["Navigate left from Home instead of right."])

# 1. Nothing recorded yet.
assert ls.open_lessons(run_dir, SID) == []

# 2. Failure is recorded with the step, screen text and navigation trace.
lesson = ls.record_failure(run_dir, SID, rca=rca, execution=execution,
                           verification=verification, run_id="run-A", run_dir=run_dir)
assert lesson and lesson["status"] == "open" and lesson["occurrences"] == 1
assert lesson["failed_step"]["action"] == "guided_navigate_to_text"
assert lesson["failed_step"]["screen_text"] == "Parties & chats"
assert "Home -> People -> Parties & chats" in lesson["navigation_notes"][0], lesson["navigation_notes"]
store = root / "lessons" / f"{SID}.json"
assert store.is_file(), "lessons must live at artifacts/lessons/, shared across runs"

# 3. A DIFFERENT run (new run dir) sees it - that is the cross-run part.
run_b = root / "runs" / "run-B"
(run_b / "reports").mkdir(parents=True)
seen = ls.open_lessons(run_b, SID)
assert [l["id"] for l in seen] == [lesson["id"]]

# 4. The same mistake again bumps occurrences instead of duplicating.
again = ls.record_failure(run_b, SID, rca=rca, execution=execution,
                          verification=verification, run_id="run-B", run_dir=run_b)
assert again["id"] == lesson["id"] and again["occurrences"] == 2
assert len(json.loads(store.read_text())["lessons"]) == 1

# 5. The planner prompt really contains the lesson.
template = (Path(__file__).resolve().parents[2] / "config" / "prompts" / "planner.j2").read_text(encoding="utf-8")
rendered = jinja2.Environment().from_string(template).render(
    lessons=ls.prompt_view(ls.open_lessons(run_b, SID)), is_replan=False,
    tools=[], controls={}, scenario={}, timings={}, max_steps=50, state={}, runtime={})
assert "LESSONS FROM EARLIER FAILED RUNS" in rendered
# tojson escapes '&' as \u0026, so match the unescaped parts.
assert "Right x3 from Home lands on Parties" in rendered
assert "Navigate left from Home" in rendered
assert "2x, last in run-B" in rendered

# 6. A rig fault is stored but never steers the planner.
rig = rca.model_copy(update={"failure_class": FailureClass.RIG_FAULT,
                             "primary_cause": "Capture card dropped signal"})
rig_exec = execution.model_copy(update={"abort_reason": "capture lost"})
ls.record_failure(run_b, SID, rca=rig, execution=rig_exec, run_id="run-B")
assert len(json.loads(store.read_text())["lessons"]) == 2
assert all(l["failure_class"] != "rig_fault" for l in ls.open_lessons(run_b, SID))

# 7. A PASS resolves open lessons; the next plan sees none.
assert ls.resolve_all(run_b, SID, "run-C") == 2
assert ls.open_lessons(run_b, SID) == []
assert all(l["status"] == "resolved" for l in json.loads(store.read_text())["lessons"])

# 8. A corrupt store never crashes a run.
store.write_text("{not json")
assert ls.open_lessons(run_b, SID) == []
assert ls.record_failure(run_b, SID, rca=rca, execution=execution) is not None

# ---- Shared lessons by UI-surface tag --------------------------------------
root2 = Path(tempfile.mkdtemp())
ga = root2 / "runs" / "run-1"
(ga / "reports").mkdir(parents=True)
GUIDE_TAGS = ["dashboard", "guide", "game-activity"]

# 9. A guide-tagged failure is copied to the shared store.
shared_lesson = ls.record_failure(ga, "entry-point-game-activity", rca=rca,
                                  execution=execution, run_id="run-1",
                                  tags=GUIDE_TAGS)
assert shared_lesson["shared_tags"] == ["dashboard", "guide"]
shared_file = root2 / "lessons" / "_shared.json"
assert shared_file.is_file()

# 10. A NEW scenario with a matching tag receives it on its very first plan.
new = ls.open_lessons(ga, "guide-menu", tags=["smoke", "guide", "overlay"])
assert len(new) == 1 and new[0]["scope"] == "shared"
assert new[0]["origin_scenario"] == "entry-point-game-activity"
view = ls.prompt_view(new)[0]
assert view["scope"] == "shared" and view["from_scenario"] == "entry-point-game-activity"

# 11. A game scenario (no surface tag) does NOT receive it.
assert ls.open_lessons(ga, "dungeons-mission-inplay",
                       tags=["xbox", "dungeons", "gameplay"]) == []

# 12. The origin scenario sees it once (own copy), not twice.
origin = ls.open_lessons(ga, "entry-point-game-activity", tags=GUIDE_TAGS)
assert [l["scope"] for l in origin] == ["scenario"]

# 13. Another scenario passing does NOT resolve a lesson it received.
assert ls.resolve_all(ga, "guide-menu", "run-2") == 0
assert len(ls.open_lessons(ga, "guide-menu", tags=["guide"])) == 1

# 14. The origin passing resolves it everywhere.
assert ls.resolve_all(ga, "entry-point-game-activity", "run-3") == 1
assert ls.open_lessons(ga, "guide-menu", tags=["guide"]) == []
assert json.loads(shared_file.read_text())["lessons"][0]["status"] == "resolved"

# 15. Rig faults are never shared.
ls.record_failure(ga, "entry-point-game-activity", rca=rig, execution=rig_exec,
                  run_id="run-4", tags=GUIDE_TAGS)
assert all(l["failure_class"] != "rig_fault"
           for l in json.loads(shared_file.read_text())["lessons"])

print("lesson store tests passed (15 cases)")
