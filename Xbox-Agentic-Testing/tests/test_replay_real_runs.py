"""Replay real hardware runs through the code that interprets them.

tests/fixtures/runs/ holds the JSON (no frames) of two real runs:
  run-20261015-125710  PASS  41/41 steps   (entry-point-game-activity)
  run-20261015-113636  FAIL  5/6 criteria  aborted by the dashboard guard

Every assertion is a fact observed on the console. A code change that makes
the framework re-read those facts differently - wrong counts, a schema change
that no longer loads old artifacts, a report or lesson that drifts - fails here
before it reaches hardware.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from core import lesson_store, scenario_report, trends_report
from core.schemas import ExecutionResult, RootCauseAnalysis, VerificationResult
from core.schemas import TestPlan as Plan  # aliased: pytest collects names starting "Test"

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "runs"
PASS_RUN = "run-20261015-125710"
FAIL_RUN = "run-20261015-113636"


def _json(run: str, name: str) -> dict:
    return json.loads((FIXTURES / run / "reports" / name).read_text(encoding="utf-8"))


@pytest.fixture()
def runs_dir(tmp_path: Path) -> Path:
    """A writable copy laid out like artifacts/ (runs/<id>/...)."""
    dst = tmp_path / "artifacts" / "runs"
    shutil.copytree(FIXTURES, dst)
    return dst


# --- schemas still load real artifacts ----------------------------------------
@pytest.mark.parametrize("run", [PASS_RUN, FAIL_RUN])
def test_real_artifacts_validate_against_current_schemas(run: str) -> None:
    ExecutionResult.model_validate(_json(run, "execution.json"))
    VerificationResult.model_validate(_json(run, "verification.json"))
    Plan.model_validate(_json(run, "plan-r1.json"))


def test_pass_run_facts() -> None:
    ex = ExecutionResult.model_validate(_json(PASS_RUN, "execution.json"))
    ver = VerificationResult.model_validate(_json(PASS_RUN, "verification.json"))
    assert ver.verdict.value == "pass" and ver.passed
    assert len(ex.steps) == 41 and not ex.aborted
    assert ex.observed_any_change
    # A PASS must carry observational proof (schema validator enforces it).
    assert any(e.is_proof for e in ver.evidence) or any(
        e.is_proof for c in ver.criteria for e in c.evidence)


def test_fail_run_facts() -> None:
    ex = ExecutionResult.model_validate(_json(FAIL_RUN, "execution.json"))
    ver = VerificationResult.model_validate(_json(FAIL_RUN, "verification.json"))
    assert ver.verdict.value == "fail"
    assert sum(c.met for c in ver.criteria) == 5 and len(ver.criteria) == 6
    assert ex.aborted and "Blind tapping prevented" in (ex.abort_reason or "")


# --- report regeneration ------------------------------------------------------
@pytest.mark.parametrize("run,verdict,passed", [(PASS_RUN, "PASS", 5), (FAIL_RUN, "FAIL", 5)])
def test_report_rebuilds_from_real_run(runs_dir: Path, run: str, verdict: str, passed: int) -> None:
    written = scenario_report.build_report(runs_dir / run)
    payload = json.loads(Path(written["json"]).read_text(encoding="utf-8"))
    assert payload["verdict"] == verdict
    assert payload["counts"].get("PASS", 0) >= passed
    html = Path(written["dashboard"]).read_text(encoding="utf-8")
    assert "Step timeline" in html and "Most time:" in html
    assert "Guided navigation" in html


def test_fail_report_shows_learning_section(runs_dir: Path) -> None:
    html = Path(scenario_report.build_report(runs_dir / FAIL_RUN)["dashboard"]).read_text(
        encoding="utf-8")
    assert "Failure learning" in html and "Recorded from this run" in html


# --- lessons from the real failure ------------------------------------------
def test_lesson_from_real_failure_points_at_the_abort_cause(runs_dir: Path) -> None:
    run_dir = runs_dir / FAIL_RUN
    lesson = lesson_store.record_failure(
        run_dir, "entry-point-game-activity",
        rca=RootCauseAnalysis.model_validate(_json(FAIL_RUN, "rca.json")),
        execution=ExecutionResult.model_validate(_json(FAIL_RUN, "execution.json")),
        verification=VerificationResult.model_validate(_json(FAIL_RUN, "verification.json")),
        run_id=FAIL_RUN, run_dir=run_dir, tags=["dashboard", "guide"])
    assert lesson is not None
    # The B press at step 31 caused the abort at 32, not the recovered step-6 error.
    assert lesson["failed_step"]["index"] == 31
    assert lesson["failed_step"]["arguments"] == {"button": "b"}
    assert lesson["shared_tags"] == ["dashboard", "guide"]


# --- trends across both runs --------------------------------------------------
def test_trends_over_real_runs(runs_dir: Path) -> None:
    data = trends_report.collect(runs_dir.parent)
    runs = data["entry-point-game-activity"]
    assert [r["verdict"] for r in runs] == ["FAIL", "PASS"]
    assert runs[0]["met"] == 5 and runs[0]["total"] == 6
    assert runs[1]["met"] == runs[1]["total"]
