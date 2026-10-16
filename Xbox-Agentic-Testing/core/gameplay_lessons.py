"""
gameplay_lessons.py - persistent learning for closed-loop gameplay runs.

lesson_store.py learns from planned console scenarios through RCA. Gameplay
runs (Dungeons, ...) never go through a planner, so they learn here instead,
from the per-cycle evidence the gameplay loop already saves.

A game supplies a rules object (see tools/game_profiles/dungeons_learning.py):

    detect(payload, report)            -> findings this run proved (a finding whose
                                          signature matches an open lesson = recurred)
    assess(lesson, payload, report)    -> "fixed" | "untested" | "quiet" (lessons
                                          that did NOT recur this run)
    adjustments(lessons)               -> bounded runtime knobs (counter overrides)
    prompt_text(lessons)               -> hint lines for the vision model

Lifecycle of a gameplay lesson (artifacts/lessons/gameplay-<game>.json):

    run start   open_lessons()  -> adjustments + prompt hints are applied
    run end     update()        -> new findings recorded, repeats bumped,
                                   fixed ones resolved, quiet ones aged/expired

The file uses the same fields as lesson_store, so `xbox-agentic lessons list`
and `lessons resolve gameplay-<game> <id>` work on it unchanged. Every public
function logs and swallows its own errors: learning must never kill a run.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, Protocol

from core.store_lock import locked

log = logging.getLogger("gameplay_lessons")

MAX_STORED = 30


class GameplayRules(Protocol):
    def detect(self, payload: dict[str, Any],
               report: dict[str, Any]) -> list[dict[str, Any]]: ...

    def assess(self, lesson: dict[str, Any], payload: dict[str, Any],
               report: dict[str, Any]) -> str: ...   # "fixed" | "untested" | "quiet"

    def adjustments(self, lessons: list[dict[str, Any]]) -> dict[str, Any]: ...

    def prompt_text(self, lessons: list[dict[str, Any]]) -> str: ...


def store_id(game: str) -> str:
    return f"gameplay-{re.sub(r'[^A-Za-z0-9._-]+', '-', game or 'unknown')}"


def lessons_path(artifacts_root: Path | str, game: str) -> Path:
    return Path(artifacts_root) / "lessons" / f"{store_id(game)}.json"


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _load(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return list(data.get("lessons") or []) if isinstance(data, dict) else []
    except Exception as exc:
        log.warning("Unreadable gameplay lesson file %s: %s", path, exc)
        return []


def _save(path: Path, game: str, lessons: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(
        {"scenario_id": store_id(game), "updated_at": _now(), "lessons": lessons},
        indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    tmp.replace(path)


def open_lessons(artifacts_root: Path | str, game: str) -> list[dict[str, Any]]:
    """Open lessons for this game, most recent first."""
    try:
        lessons = [l for l in _load(lessons_path(artifacts_root, game))
                   if l.get("status") == "open"]
        return sorted(lessons, key=lambda l: str(l.get("last_seen_at", "")), reverse=True)
    except Exception as exc:
        log.warning("Could not read gameplay lessons for %s: %s", game, exc)
        return []


def _update_unlocked(artifacts_root: Path | str, game: str, rules: GameplayRules,
                     payload: dict[str, Any], report: dict[str, Any],
                     applied_ids: list[str], run_id: str,
                     expire_after: int) -> dict[str, Any]:
    path = lessons_path(artifacts_root, game)
    lessons = _load(path)
    out: dict[str, Any] = {"recorded": [], "recurred": [], "resolved": [],
                           "expired": [], "untested": [], "quiet": []}
    applied = set(applied_ids or [])
    findings = rules.detect(payload, report)
    detected = {str(f.get("signature")) for f in findings}

    # 1. Judge every open lesson that did not recur against this run's evidence.
    for lesson in lessons:
        if lesson.get("status") != "open" or str(lesson.get("signature")) in detected:
            continue
        verdict = rules.assess(lesson, payload, report)
        lid = str(lesson.get("id"))
        if verdict == "fixed":
            lesson.update(status="resolved", resolved_at=_now(), resolved_by_run=run_id,
                          resolution=lesson.get("_fix_note") or "did not recur; the "
                          "scenario it blocked now passes")
            lesson.pop("_fix_note", None)
            out["resolved"].append(lid)
        elif verdict == "quiet" and lid in applied:
            lesson["quiet_runs"] = int(lesson.get("quiet_runs", 0)) + 1
            if expire_after > 0 and lesson["quiet_runs"] >= expire_after:
                lesson.update(status="expired", resolved_at=_now(), resolved_by_run=run_id,
                              resolution=f"did not recur in {lesson['quiet_runs']} runs")
                out["expired"].append(lid)
            else:
                out["quiet"].append(lid)
        elif verdict == "untested":
            out["untested"].append(lid)

    # 2. Record what this run proved went wrong (repeats bump the open lesson).
    for finding in findings:
        sig = str(finding.get("signature"))
        existing = next((l for l in lessons if l.get("signature") == sig
                         and l.get("status") == "open"), None)
        if existing:
            existing["occurrences"] = int(existing.get("occurrences", 1)) + 1
            existing["quiet_runs"] = 0
            existing["last_seen_at"] = _now()
            existing["last_run_id"] = run_id
            existing["evidence"] = finding.get("evidence")
            existing["primary_cause"] = finding.get("primary_cause")
            existing["data"] = {**(existing.get("data") or {}), **(finding.get("data") or {})}
            out["recurred"].append(str(existing.get("id")))
            continue
        lesson = {
            "id": f"G{len(lessons) + 1:03d}-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "status": "open",
            "applies_to_plan": True,
            "signature": sig,
            "created_at": _now(),
            "last_seen_at": _now(),
            "run_id": run_id,
            "last_run_id": run_id,
            "occurrences": 1,
            "quiet_runs": 0,
            "failure_class": finding.get("rule"),
            "scenario": finding.get("scenario"),
            "primary_cause": finding.get("primary_cause"),
            "evidence": finding.get("evidence"),
            "data": finding.get("data") or {},
        }
        lessons.append(lesson)
        out["recorded"].append(lesson["id"])

    if len(lessons) > MAX_STORED:
        closed = [l for l in lessons if l.get("status") != "open"]
        for old in closed[:len(lessons) - MAX_STORED]:
            lessons.remove(old)
    _save(path, game, lessons)
    by_id = {str(l.get("id")): l for l in lessons}
    out["lessons"] = {k: [by_id[i] for i in v if i in by_id] for k, v in out.items()}
    return out


def update(artifacts_root: Path | str, game: str, rules: GameplayRules,
           payload: dict[str, Any], report: dict[str, Any],
           applied_ids: list[str], run_id: str = "",
           expire_after: int = 3) -> dict[str, Any]:
    """Record, bump, resolve and age lessons from one finished gameplay run."""
    try:
        with locked(lessons_path(artifacts_root, game)):
            return _update_unlocked(artifacts_root, game, rules, payload, report,
                                    applied_ids, run_id, expire_after)
    except Exception as exc:
        log.warning("Could not update gameplay lessons for %s: %s", game, exc)
        return {"error": f"{type(exc).__name__}: {exc}"}
