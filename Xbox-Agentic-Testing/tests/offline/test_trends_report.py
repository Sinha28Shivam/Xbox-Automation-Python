"""Trends page: per-scenario history across runs, dry runs excluded."""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core import trends_report as tr

root = Path(tempfile.mkdtemp())


def _run(name, sid, verdict, met, total, dispatched, dur, lessons=()):
    rep = root / "runs" / name / "reports"
    rep.mkdir(parents=True)
    (rep / "verification.json").write_text(json.dumps({
        "scenario_id": sid, "verdict": verdict,
        "criteria": [{"met": i < met} for i in range(total)]}))
    (rep / "execution.json").write_text(json.dumps({
        "scenario_id": sid, "dispatched_steps": dispatched,
        "duration_seconds": dur, "steps": [{}] * 5}))
    if lessons:
        (rep / "lessons-applied-r1.json").write_text(json.dumps(
            {"lessons": [{"id": l} for l in lessons]}))


_run("run-1", "ga", "fail", 0, 6, 7, 60)
_run("run-2", "ga", "inconclusive", 0, 0, 0, 0)          # dry run
_run("run-3", "ga", "fail", 5, 6, 30, 334, ["L001"])
_run("run-4", "ga", "pass", 6, 6, 28, 180, ["L001"])
_run("run-5", "other", "pass", 1, 1, 3, 10)

data = tr.collect(root)
assert [r["run_id"] for r in data["ga"]] == ["run-1", "run-3", "run-4"], "dry run excluded"
assert [r["met"] for r in data["ga"]] == [0, 5, 6]
assert data["ga"][1]["lessons"] == ["L001"]

out = tr.build(root)
page = out.read_text(encoding="utf-8")
assert "pass rate 33%" in page and "Overall pass rate" in page
assert page.count("<svg") == 4, "two sparklines per scenario"
assert "L001" in page

print("trends report tests passed")
