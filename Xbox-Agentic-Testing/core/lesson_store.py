"""
lesson_store.py - persistent failure learning, per scenario.

route_store.py remembers what WORKED (a passing plan is replayed). This module
remembers what FAILED, so the planner stops repeating a mistake that an earlier
run already made - e.g. "Right x3 from Home lands on Parties & chats, not Game
Activity" (run-20261014-162340).

Lifecycle of a lesson
---------------------
    RCA after a failure        -> record_failure()   status = open
    Planner, every plan        -> open_lessons()     injected into the prompt
    Verifier on PASS           -> resolve_all()      status = resolved

Lessons live in artifacts/lessons/<scenario_id>.json, beside artifacts/routes/,
so they survive across runs. Every public function swallows its own errors and
logs them: learning is an aid, and a corrupt lesson file must never be the
reason a hardware run dies.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger("lesson_store")

# Rig and environment faults say nothing about whether the PLAN was right, so
# they are recorded for the record but never steer the planner.
NON_PLAN_CLASSES = {"rig_fault", "environment"}
MAX_STORED = 30

# Shared lessons
# --------------
# Tags naming a console UI SURFACE (not a single test) mark behaviour that any
# scenario on that surface will meet, e.g. "pressing Guide while the Guide is
# open closes it". A lesson from a scenario carrying one of these tags is also
# copied to lessons/_shared.json and offered to every other scenario that
# shares the tag. Overridable via settings learning.shared_tags.
DEFAULT_SHARED_TAGS = {"guide", "dashboard", "store", "search", "settings"}
SHARED_FILE = "_shared.json"


def _base_dir(artifacts_dir: Path | str) -> Path:
    path = Path(artifacts_dir)
    return path.parent.parent if "runs" in path.parts else path


def lessons_path(artifacts_dir: Path | str, scenario_id: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", scenario_id or "unknown")
    return _base_dir(artifacts_dir) / "lessons" / f"{safe}.json"


def shared_path(artifacts_dir: Path | str) -> Path:
    return _base_dir(artifacts_dir) / "lessons" / SHARED_FILE


def _surface_tags(tags: Any, shared_tags: Any = None) -> list[str]:
    allowed = {str(t).lower() for t in (shared_tags or DEFAULT_SHARED_TAGS)}
    return sorted({str(t).lower() for t in (tags or [])} & allowed)


def _sync_shared(artifacts_dir: Path | str, origin: str,
                 lesson: dict[str, Any], tags: list[str]) -> None:
    """Upsert the shared copy of a lesson, keyed by (origin scenario, id)."""
    path = shared_path(artifacts_dir)
    shared = _load(path)
    entry = next((l for l in shared if l.get("origin_scenario") == origin
                  and l.get("id") == lesson.get("id")), None)
    copy = {**lesson, "origin_scenario": origin, "tags": tags}
    if entry is None:
        shared.append(copy)
    else:
        entry.update(copy)
    if len(shared) > MAX_STORED * 3:
        resolved = [l for l in shared if l.get("status") != "open"]
        for old in resolved[:len(shared) - MAX_STORED * 3]:
            shared.remove(old)
    _save(path, "_shared", shared)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _load(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return list(data.get("lessons") or []) if isinstance(data, dict) else []
    except Exception as exc:
        log.warning("Unreadable lesson file %s: %s", path, exc)
        return []


def _save(path: Path, scenario_id: str, lessons: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(
        {"scenario_id": scenario_id, "updated_at": _now(), "lessons": lessons},
        indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    tmp.replace(path)


def _dump(obj: Any) -> dict[str, Any]:
    if obj is None:
        return {}
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    return dict(obj)


def _value(v: Any) -> str:
    return str(getattr(v, "value", v) or "")


_INPUT_ACTIONS = {"press_button", "hold_button", "run_sequence", "type_text",
                  "move_stick", "guided_navigate_to_text"}


def _failed_step(execution: dict[str, Any]) -> dict[str, Any]:
    """The step that CAUSED the failure, not the one that noticed it.

    When the executor refuses a step (e.g. "blind tapping prevented"), the
    refused step is innocent - the damage was done by the last input sent
    before it. That input is what the planner must change next time.
    """
    steps = execution.get("steps") or []
    if not steps:
        return {}
    # An abort ends the run at the LAST step, so that is where the decisive
    # failure is. Earlier step errors may have been recovered from: in
    # run-20261015-113636 step 6 errored but the run went on to pass 5/6
    # criteria, and the abort at step 32 was caused by step 31.
    if execution.get("aborted"):
        anchor = len(steps) - 1
    else:
        anchor = next((i for i, s in enumerate(steps) if s.get("error")), len(steps) - 1)
    chosen = steps[anchor]
    if chosen.get("action") not in _INPUT_ACTIONS or not chosen.get("dispatched", True):
        cause = next((s for s in reversed(steps[:anchor + 1])
                      if s.get("action") in _INPUT_ACTIONS and s.get("dispatched")), None)
        chosen = cause or chosen
    return {
        "index": chosen.get("index"),
        "action": chosen.get("action"),
        "arguments": chosen.get("arguments") or {},
        "error": (chosen.get("error") or "")[:400],
        "observation": (chosen.get("observation") or "")[:400],
        "screen_text": (chosen.get("ocr_text") or "")[:600],
    }


# Xbox chrome that appears on every Guide panel and says nothing about which
# panel is showing; skipping it leaves the panel heading.
_CHROME = re.compile(
    r"^(hold for (power )?options|share last capture|more options|"
    r"[\d,. :%]+(am|pm)?|[^A-Za-z]*|.{0,3})$", re.IGNORECASE)


def _screen_label(attempt: dict[str, Any]) -> str:
    if attempt.get("focused_label"):
        return str(attempt["focused_label"]).strip()[:30]
    for line in attempt.get("visible_lines") or []:
        text = line.strip()
        if not _CHROME.match(text) and sum(ch.isalpha() for ch in text) >= 4:
            return text[:30]
    return "?"


def _navigation_notes(run_dir: Path | None) -> list[str]:
    """One line per guided navigation: what was searched and what was seen."""
    if run_dir is None:
        return []
    notes: list[str] = []
    for path in sorted((Path(run_dir) / "reports").glob("guided-navigation*.json")):
        try:
            trace = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        attempts = trace.get("attempts") or []
        # Older traces have no "found" key; any matched attempt means success.
        if trace.get("found") or any(a.get("matched") for a in attempts):
            continue
        screens = [_screen_label(a) for a in attempts]
        state = ("visible but never highlighted" if trace.get("text_seen")
                 else "never appeared")
        direction = trace.get("direction") or (attempts[0].get("direction") if attempts else "?")
        notes.append(
            f"Searching '{trace.get('target_text')}' by pressing {direction} "
            f"{max(0, len(attempts) - 1)}x: target {state}. Screens seen in order: "
            + " -> ".join(screens))
    return notes[:5]


def _signature(failure_class: str, step: dict[str, Any], abort_reason: str) -> str:
    """Stable identity for 'the same mistake', independent of LLM wording."""
    args = json.dumps(step.get("arguments") or {}, sort_keys=True)
    reason = re.sub(r"\d+", "#", (abort_reason or step.get("error") or ""))[:120]
    return f"{failure_class}|{step.get('action')}|{args}|{reason}".lower()


def record_failure(artifacts_dir: Path | str, scenario_id: str, *,
                   rca: Any, execution: Any = None, verification: Any = None,
                   run_id: str = "", revision: int = 1,
                   run_dir: Path | str | None = None,
                   tags: Any = None,
                   shared_tags: Any = None) -> dict[str, Any] | None:
    """Store what went wrong. Repeats of the same mistake bump `occurrences`.

    When the scenario carries a UI-surface tag (guide, dashboard, ...), the
    lesson is also copied to the shared store for other scenarios on that
    surface.
    """
    try:
        rca_d, ex_d, ver_d = _dump(rca), _dump(execution), _dump(verification)
        failure_class = _value(rca_d.get("failure_class")) or "unknown"
        step = _failed_step(ex_d)
        abort_reason = ex_d.get("abort_reason") or ""
        signature = _signature(failure_class, step, abort_reason)

        path = lessons_path(artifacts_dir, scenario_id)
        lessons = _load(path)
        existing = next((l for l in lessons
                         if l.get("signature") == signature and l.get("status") == "open"), None)
        surface = _surface_tags(tags, shared_tags)
        if existing:
            existing["occurrences"] = int(existing.get("occurrences", 1)) + 1
            existing["last_seen_at"] = _now()
            existing["last_run_id"] = run_id
            _save(path, scenario_id, lessons)
            if surface and existing.get("applies_to_plan", True):
                _sync_shared(artifacts_dir, scenario_id, existing, surface)
            return existing

        lesson = {
            "id": f"L{len(lessons) + 1:03d}-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "status": "open",
            "applies_to_plan": failure_class not in NON_PLAN_CLASSES,
            "signature": signature,
            "created_at": _now(),
            "last_seen_at": _now(),
            "run_id": run_id,
            "last_run_id": run_id,
            "revision": revision,
            "occurrences": 1,
            "verdict": _value(ver_d.get("verdict")),
            "failure_class": failure_class,
            "primary_cause": (rca_d.get("primary_cause") or "")[:600],
            "abort_reason": abort_reason[:300],
            "failed_step": step,
            "not_proven": list(ver_d.get("not_proven") or [])[:4],
            "navigation_notes": _navigation_notes(Path(run_dir) if run_dir else None),
            "recommendations": list(rca_d.get("recommendations") or [])[:3],
            "shared_tags": surface,
        }
        lessons.append(lesson)
        # Keep every open lesson; trim the oldest resolved ones first.
        if len(lessons) > MAX_STORED:
            resolved = [l for l in lessons if l.get("status") != "open"]
            for old in resolved[:len(lessons) - MAX_STORED]:
                lessons.remove(old)
        _save(path, scenario_id, lessons)
        if surface and lesson["applies_to_plan"]:
            _sync_shared(artifacts_dir, scenario_id, lesson, surface)
        return lesson
    except Exception as exc:
        log.warning("Could not record lesson for %s: %s", scenario_id, exc)
        return None


def _recent_first(lessons: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(lessons, key=lambda l: (str(l.get("last_seen_at", "")),
                                          int(l.get("occurrences", 1))), reverse=True)


def open_lessons(artifacts_dir: Path | str, scenario_id: str,
                 limit: int = 5, tags: Any = None,
                 shared_tags: Any = None,
                 shared_limit: int = 3) -> list[dict[str, Any]]:
    """Unresolved, plan-relevant lessons for this scenario, then shared ones.

    Own lessons come first and carry scope="scenario". Shared lessons from
    OTHER scenarios on the same UI surface follow with scope="shared"; the
    origin scenario already has its own copy, so it is never offered twice.
    """
    try:
        own = _recent_first([
            {**l, "scope": "scenario"}
            for l in _load(lessons_path(artifacts_dir, scenario_id))
            if l.get("status") == "open" and l.get("applies_to_plan", True)])
        own = own[:max(0, int(limit))]

        surface = set(_surface_tags(tags, shared_tags))
        shared: list[dict[str, Any]] = []
        if surface and shared_limit > 0:
            own_sigs = {l.get("signature") for l in own}
            shared = _recent_first([
                {**l, "scope": "shared"}
                for l in _load(shared_path(artifacts_dir))
                if l.get("status") == "open"
                and l.get("origin_scenario") != scenario_id
                and surface & set(l.get("tags") or [])
                and l.get("signature") not in own_sigs])
            shared = shared[:max(0, int(shared_limit))]
        return own + shared
    except Exception as exc:
        log.warning("Could not read lessons for %s: %s", scenario_id, exc)
        return []


def resolve_all(artifacts_dir: Path | str, scenario_id: str,
                run_id: str = "") -> int:
    """A PASS proves this scenario's open lessons were overcome.

    Only the scenario's OWN lessons (and their shared copies) are resolved. A
    different scenario passing proves nothing about the origin's failure, so
    shared lessons received from elsewhere stay open.
    """
    try:
        path = lessons_path(artifacts_dir, scenario_id)
        lessons = _load(path)
        resolved_ids = set()
        for lesson in lessons:
            if lesson.get("status") == "open":
                lesson["status"] = "resolved"
                lesson["resolved_at"] = _now()
                lesson["resolved_by_run"] = run_id
                resolved_ids.add(lesson.get("id"))
        if resolved_ids:
            _save(path, scenario_id, lessons)
            _mark_shared(artifacts_dir, scenario_id, resolved_ids, "resolved", run_id)
        return len(resolved_ids)
    except Exception as exc:
        log.warning("Could not resolve lessons for %s: %s", scenario_id, exc)
        return 0


def note_run(artifacts_dir: Path | str, scenario_id: str,
             applied_ids: list[str], recorded_id: str | None,
             run_id: str = "", expire_after: int = 3) -> list[str]:
    """Age the scenario's open lessons; expire ones that stopped recurring.

    Called once per finished (non-dry) run. A lesson that was applied but did
    NOT happen again this run gets `quiet_runs += 1`; after `expire_after`
    such runs it is closed as "expired". This stops a scenario that never
    fully passes (for an unrelated reason) from carrying fixed problems
    forever. A recurrence resets the counter.
    """
    try:
        path = lessons_path(artifacts_dir, scenario_id)
        lessons = _load(path)
        expired: list[str] = []
        changed = False
        for lesson in lessons:
            if lesson.get("status") != "open":
                continue
            if lesson.get("id") == recorded_id:
                lesson["quiet_runs"] = 0
                changed = True
                continue
            if lesson.get("id") not in set(applied_ids or []):
                continue
            lesson["quiet_runs"] = int(lesson.get("quiet_runs", 0)) + 1
            changed = True
            if expire_after > 0 and lesson["quiet_runs"] >= expire_after:
                lesson.update(status="expired", resolved_at=_now(),
                              resolved_by_run=run_id,
                              resolution=f"did not recur in {lesson['quiet_runs']} runs")
                expired.append(str(lesson.get("id")))
        if changed:
            _save(path, scenario_id, lessons)
        if expired:
            _mark_shared(artifacts_dir, scenario_id, set(expired), "expired", run_id)
        return expired
    except Exception as exc:
        log.warning("Could not age lessons for %s: %s", scenario_id, exc)
        return []


def resolve_one(artifacts_dir: Path | str, scenario_id: str, lesson_id: str,
                reason: str = "resolved manually") -> bool:
    """Close one lesson by hand, e.g. after the bug it describes was fixed in code."""
    path = lessons_path(artifacts_dir, scenario_id)
    lessons = _load(path)
    hit = next((l for l in lessons if str(l.get("id", "")).startswith(lesson_id)
                and l.get("status") == "open"), None)
    if hit is None:
        return False
    hit.update(status="resolved", resolved_at=_now(), resolved_by_run="manual",
               resolution=reason)
    _save(path, scenario_id, lessons)
    _mark_shared(artifacts_dir, scenario_id, {hit["id"]}, "resolved", "manual")
    return True


def _mark_shared(artifacts_dir: Path | str, scenario_id: str, ids: set[str],
                 status: str, run_id: str) -> None:
    spath = shared_path(artifacts_dir)
    shared = _load(spath)
    touched = False
    for entry in shared:
        if entry.get("origin_scenario") == scenario_id and entry.get("id") in ids:
            entry.update(status=status, resolved_at=_now(), resolved_by_run=run_id)
            touched = True
    if touched:
        _save(spath, "_shared", shared)


def all_lessons(artifacts_dir: Path | str) -> dict[str, list[dict[str, Any]]]:
    """Every scenario's lesson file, for the CLI listing."""
    folder = _base_dir(artifacts_dir) / "lessons"
    out: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(folder.glob("*.json")) if folder.is_dir() else []:
        if path.name == SHARED_FILE:
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            out[str(data.get("scenario_id") or path.stem)] = list(data.get("lessons") or [])
        except Exception:
            continue
    return out


def prompt_view(lessons: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The compact form the planner sees: no signatures or bookkeeping."""
    view = []
    for l in lessons:
        step = l.get("failed_step") or {}
        view.append({
            "id": l.get("id"),
            "scope": l.get("scope", "scenario"),
            "from_scenario": l.get("origin_scenario") if l.get("scope") == "shared" else None,
            "surface_tags": l.get("tags") if l.get("scope") == "shared" else None,
            "seen": f"{l.get('occurrences', 1)}x, last in {l.get('last_run_id')}",
            "failure_class": l.get("failure_class"),
            "what_went_wrong": l.get("primary_cause"),
            "abort_reason": l.get("abort_reason"),
            "failed_step": {k: step.get(k) for k in ("index", "action", "arguments", "error")},
            "screen_text_at_failure": (step.get("screen_text") or "")[:300],
            "navigation_observed": l.get("navigation_notes"),
            "recommendations": l.get("recommendations"),
        })
    return view
