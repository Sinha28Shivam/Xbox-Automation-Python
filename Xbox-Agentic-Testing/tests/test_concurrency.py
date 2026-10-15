"""Concurrent writers must not lose each other's updates or share a run folder.

Uses real OS processes (not threads), because the risk is two console runs -
e.g. a suite plus a manual run - writing the same artifacts/ store.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

WRITERS = 6
EACH = 5


def _record(root: str, worker: int) -> None:
    sys.path.insert(0, str(PROJECT))
    from core import lesson_store
    from core.schemas import ExecutionResult, FailureClass, RootCauseAnalysis, StepResult
    for i in range(EACH):
        # Distinct failing step per write -> distinct signature -> new lesson.
        ex = ExecutionResult(scenario_id="s", steps=[StepResult(
            index=i, action="press_button", dispatched=True,
            arguments={"button": "a", "w": worker, "i": i}, error="x")])
        lesson_store.record_failure(
            root, "s", execution=ex, run_id=f"w{worker}",
            rca=RootCauseAnalysis(scenario_id="s", primary_cause=f"w{worker}-{i}",
                                  failure_class=FailureClass.AUTOMATION_DEFECT),
            tags=["guide"])


def _claim(root: str, out: mp.Queue[str]) -> None:
    sys.path.insert(0, str(PROJECT))
    from core.artifacts import new_run_id
    for _ in range(3):
        out.put(new_run_id(root=root))


def test_parallel_lesson_writes_lose_nothing(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    procs = [mp.Process(target=_record, args=(str(root), w)) for w in range(WRITERS)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(60)
        assert p.exitcode == 0
    own = json.loads((root / "lessons" / "s.json").read_text(encoding="utf-8"))["lessons"]
    shared = json.loads((root / "lessons" / "_shared.json").read_text(encoding="utf-8"))["lessons"]
    causes = {l["primary_cause"] for l in own}
    assert len(causes) == WRITERS * EACH, f"lost updates: {WRITERS * EACH - len(causes)}"
    assert len(shared) == WRITERS * EACH
    assert not list((root / "lessons").glob("*.tmp")), "no stray tmp files"


def test_run_ids_unique_under_contention(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    queue: mp.Queue[str] = mp.Queue()
    procs = [mp.Process(target=_claim, args=(str(root), queue)) for _ in range(4)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(60)
    ids = [queue.get(timeout=5) for _ in range(12)]
    assert len(set(ids)) == 12, ids
    assert all((root / "runs" / i).is_dir() for i in ids)
