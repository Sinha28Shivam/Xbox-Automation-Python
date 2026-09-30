"""skill_memory.py - a persistent, (profile, skill_name)-keyed journal of
VERIFIED move sequences, so a proven answer to "how do I do X" is reused
instead of re-derived from a blank frame every single cycle.

WHY THIS EXISTS
----------------
location_memory.py already proved the pattern: a coordinate-keyed journal
that survives across runs so "where is the village" does not need
re-discovering every session. This module is the same pattern applied to
MOVE SEQUENCES instead of PLACES: "what combo reliably breaks a log" should
not need re-deciding from scratch every cycle either, once it has been
observed to work.

WHY KEYED BY SCENE, NOT BY GOAL
---------------------------------
The overall goal ("find a tree, craft planks") is fixed for an entire run
and too coarse to key a reusable skill on. What actually repeats is a
SITUATION - a particular scene_state or condition (e.g. "facing a log,
close enough to mine") - which is exactly what the vision model already
classifies every cycle. Keying skills on that classification means a
lookup can happen right after the model reads the frame, before the model's
own (possibly low-confidence) move choice is trusted.

WHAT THIS DOES NOT DO
------------------------
It does not skip the vision call - the model still looks at every frame and
still classifies the scene (that classification is what decides whether a
skill even applies). What it skips is re-deciding the MOVE PARAMETERS for a
situation that has already been solved and measured to work, and it uses a
win-rate (successes/attempts) so a sequence that stops working (e.g. because
the game state around it changed) gets abandoned rather than replayed
forever on faith.

PERSISTENCE ACROSS RUNS
------------------------
Same top-level artifacts/ location as world_memory.json (location_memory.py)
and artifacts/routes/ (route_store.py) - a skill learned in one session must
be recallable in the next.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from registry import ToolContext, ToolSpec, fail, make_tool, ok


def _memory_path(ctx: ToolContext) -> Path:
    run_dir = ctx.artifacts.run_dir
    base_dir = run_dir.parent.parent if "runs" in run_dir.parts else run_dir
    return base_dir / "skill_memory.json"


def _load(ctx: ToolContext) -> list[dict[str, Any]]:
    path = _memory_path(ctx)
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _save(ctx: ToolContext, entries: list[dict[str, Any]]) -> None:
    path = _memory_path(ctx)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")


def _find_entry(entries: list[dict[str, Any]], profile: str,
                skill_name: str) -> dict[str, Any] | None:
    for e in entries:
        if e.get("profile") == profile and e.get("skill_name") == skill_name:
            return e
    return None


# ===========================================================================
# Record / update a skill
# ===========================================================================
def record_skill_impl(
    ctx: ToolContext,
    profile: str,
    skill_name: str,
    moves: list[dict[str, Any]],
    notes: str = "",
) -> dict[str, Any]:
    """Save a move sequence as the current best-known answer for `skill_name`.

    Overwrites any EXISTING sequence for the same (profile, skill_name) and
    resets its win-rate counters to 0 - the new sequence is an untested
    replacement, not a continuation of the old one's track record.
    """
    if not moves:
        return fail("A skill needs at least one move.", skill_name=skill_name)

    entries = _load(ctx)
    existing = _find_entry(entries, profile, skill_name)
    now = datetime.now(timezone.utc).isoformat()
    entry = {
        "profile": profile,
        "skill_name": skill_name,
        "moves": moves,
        "notes": notes,
        "created_at": existing["created_at"] if existing else now,
        "last_used_at": now,
        "attempts": 0,
        "successes": 0,
        "success_rate": None,
    }
    if existing:
        entries[entries.index(existing)] = entry
    else:
        entries.append(entry)
    _save(ctx, entries)

    return ok(entry=entry, total_skills=len(entries),
             journal_path=str(_memory_path(ctx)))


def _record_skill(ctx: ToolContext) -> Any:
    def run(profile: str, skill_name: str, moves: list[dict[str, Any]],
            notes: str = "") -> dict[str, Any]:
        return record_skill_impl(ctx, profile=profile, skill_name=skill_name,
                                 moves=moves, notes=notes)

    return make_tool(
        run, "record_skill",
        "Save a move sequence (list of {action, direction, duration, "
        "button} dicts) as the current best-known way to handle a named "
        "situation (e.g. 'mine_log', 'climb_ledge') for a given game "
        "profile. Overwrites any previous sequence for the same name and "
        "resets its win-rate to untested.")


# ===========================================================================
# Recall a skill
# ===========================================================================
def find_skill_impl(ctx: ToolContext, profile: str,
                    skill_name: str) -> dict[str, Any]:
    entries = _load(ctx)
    entry = _find_entry(entries, profile, skill_name)
    if entry is None:
        return fail(f"No skill named '{skill_name}' for profile '{profile}'.",
                    profile=profile, skill_name=skill_name)
    return ok(entry=entry, journal_path=str(_memory_path(ctx)))


def _find_skill(ctx: ToolContext) -> Any:
    def run(profile: str, skill_name: str) -> dict[str, Any]:
        return find_skill_impl(ctx, profile=profile, skill_name=skill_name)

    return make_tool(
        run, "find_skill",
        "Recall the saved move sequence for a named situation under a "
        "given game profile, including its attempts/successes/success_rate "
        "so far. Fails if no skill with that name has been recorded yet.")


def list_skills_impl(ctx: ToolContext, profile: str = "") -> dict[str, Any]:
    entries = _load(ctx)
    if profile:
        entries = [e for e in entries if e.get("profile") == profile]
    ranked = sorted(entries, key=lambda e: (e.get("success_rate") or 0.0),
                    reverse=True)
    return ok(skills=ranked, total=len(ranked),
             journal_path=str(_memory_path(ctx)))


def _list_skills(ctx: ToolContext) -> Any:
    def run(profile: str = "") -> dict[str, Any]:
        return list_skills_impl(ctx, profile=profile)

    return make_tool(
        run, "list_skills",
        "List every saved skill, optionally filtered to one game profile, "
        "ranked by success_rate descending.")


# ===========================================================================
# Report an outcome - the win-rate is what keeps a skill honest
# ===========================================================================
def report_skill_outcome_impl(ctx: ToolContext, profile: str, skill_name: str,
                              success: bool) -> dict[str, Any]:
    """Record whether replaying this skill actually worked THIS time.

    A skill's stored moves are only ever as good as the last time they were
    checked against a measured delta - this is what lets a sequence that
    stops working (e.g. the game state around it changed) get abandoned by
    its own dropping success_rate instead of being replayed forever.
    """
    entries = _load(ctx)
    entry = _find_entry(entries, profile, skill_name)
    if entry is None:
        return fail(f"No skill named '{skill_name}' for profile '{profile}'.",
                    profile=profile, skill_name=skill_name)

    entry["attempts"] = int(entry.get("attempts", 0)) + 1
    entry["successes"] = int(entry.get("successes", 0)) + (1 if success else 0)
    entry["success_rate"] = round(entry["successes"] / entry["attempts"], 3)
    entry["last_used_at"] = datetime.now(timezone.utc).isoformat()
    _save(ctx, entries)

    return ok(entry=entry, journal_path=str(_memory_path(ctx)))


def _report_skill_outcome(ctx: ToolContext) -> Any:
    def run(profile: str, skill_name: str, success: bool) -> dict[str, Any]:
        return report_skill_outcome_impl(ctx, profile=profile,
                                         skill_name=skill_name, success=success)

    return make_tool(
        run, "report_skill_outcome",
        "Record whether replaying a named skill's saved moves actually "
        "worked this time (measured, e.g. by pixel delta) - updates its "
        "attempts/successes/success_rate so a sequence that stops working "
        "loses trust instead of being replayed on faith forever.")


def provide() -> list[ToolSpec]:
    return [
        ToolSpec(name="record_skill",
                 description="Save a verified move sequence for a named situation.",
                 tags=["analysis", "memory"],
                 factory=_record_skill, mutates_hardware=False),
        ToolSpec(name="find_skill",
                 description="Recall a saved move sequence and its win-rate.",
                 tags=["analysis", "memory"],
                 factory=_find_skill, mutates_hardware=False),
        ToolSpec(name="list_skills",
                 description="List every saved skill, ranked by win-rate.",
                 tags=["analysis", "memory"],
                 factory=_list_skills, mutates_hardware=False),
        ToolSpec(name="report_skill_outcome",
                 description="Record whether a replayed skill worked this time.",
                 tags=["analysis", "memory"],
                 factory=_report_skill_outcome, mutates_hardware=False),
    ]
