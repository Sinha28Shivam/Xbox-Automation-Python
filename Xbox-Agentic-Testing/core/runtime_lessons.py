"""
runtime_lessons.py - lessons enforced DURING execution, not just at planning.

A plan is a fixed step list with no conditionals. The planner can promise
"check the Guide state before opening it" (run-20261015-113636 did, citing
L001) and still emit an unconditional `press_button guide` - so whether the
lesson actually protects the run depended on luck. This module closes that
gap: immediately before an input step is dispatched, the executor reads the
LIVE screen and asks each rule whether the step is still safe.

Two kinds of rule
-----------------
1. Built-in screen rules, triggered by a lesson's content:
     guide_already_open - skip `press_button guide` when the Guide overlay is
                          already on screen (pressing it would CLOSE it).
2. Generic repeat check: an input step identical (action + arguments) to an
   open lesson's failing step is never blocked - it may be correct in this
   context - but the decision is recorded with the screen it was taken on,
   so the report can show the lesson was checked at the moment it mattered.

Every decision is a plain dict appended to ExecutionResult.notes-adjacent
`runtime_lessons` log and saved to runtime-lessons-r<N>.json.
"""

from __future__ import annotations

import re
from typing import Any

# Text that appears ONLY when the Xbox Guide overlay is open. Deliberately
# Guide-panel chrome, not words like "Achievements" that also appear on
# dashboard tiles or inside games.
GUIDE_OPEN_MARKERS = (
    "hold for power options",
    "hold for options",
    "game activity",
    "parties & chats",
    "parties and chats",
    "share last capture",
    "friend requests",
)

_GUIDE_LESSON = re.compile(
    r"guide\b.{0,80}\b(already open|was open|closed it|closes it|close it)",
    re.IGNORECASE | re.DOTALL)

# run-20261015-113636: an early A opened Achievements, then the plan pressed A
# again on the same page. Triggered by lessons about re-opening / pressing A on
# an already-open destination.
_ALREADY_OPEN_LESSON = re.compile(
    r"(pressed a (again|early|too early|twice)|press(ing)? a on .{0,30}already|"
    r"a press .{0,40}already (open|opened)|opened .{0,40} (instead|too early))",
    re.IGNORECASE)

# NOT a rule (measured, run-20261015-113636): "skip B when already on the
# dashboard". The Guide panel is translucent, so the dashboard's "SPONSORED" /
# "Game Pass" text is OCR'd behind Guide and Stats screens too; with the
# Guide's own chrome misread, 6 non-dashboard frames looked like the
# dashboard. A rule that suppresses input must not fire on evidence that weak.


def _lesson_text(lesson: dict[str, Any]) -> str:
    parts = [str(lesson.get(k) or "") for k in
             ("what_went_wrong", "primary_cause", "abort_reason")]
    parts += [str(r) for r in lesson.get("recommendations") or []]
    return " ".join(parts)


def active_rules(lessons: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Which built-in rules the open lessons switch on, and by which lesson ids."""
    rules: dict[str, list[str]] = {}
    for lesson in lessons or []:
        text, lid = _lesson_text(lesson), str(lesson.get("id"))
        if _GUIDE_LESSON.search(text):
            rules.setdefault("guide_already_open", []).append(lid)
        if _ALREADY_OPEN_LESSON.search(text):
            rules.setdefault("destination_already_open", []).append(lid)
    return rules


def _is_press(action: str, arguments: dict[str, Any], *buttons: str) -> bool:
    return (action == "press_button"
            and str(arguments.get("button", "")).lower() in buttons)


def _destination_heading(screen_text: str, targets: list[str]) -> str | None:
    """The first real OCR line, if it is one of the scenario's destinations.

    Callers must pass only LEAF destinations (pages that are opened, not
    containers that hold the selectable items). Measured on
    run-20261015-113636: the Game Activity panel heads every A press that
    opens Achievements/Stats/club, so including it would block every one.
    """
    from core.text_match import normalise
    for line in (screen_text or "").splitlines():
        clean = line.strip()
        if sum(ch.isalpha() for ch in clean) < 3:
            continue
        head = normalise(clean)
        return next((t for t in targets if normalise(t) == head), None)
    return None


def guide_is_open(screen_text: str) -> str | None:
    low = (screen_text or "").lower()
    return next((m for m in GUIDE_OPEN_MARKERS if m in low), None)


def _is_guide_press(action: str, arguments: dict[str, Any]) -> bool:
    return (action == "press_button"
            and str(arguments.get("button", "")).lower() in {"guide", "xbox", "home"}
            and int(arguments.get("times", 1) or 1) == 1)


def needs_screen(action: str, arguments: dict[str, Any],
                 rules: dict[str, list[str]],
                 lessons: list[dict[str, Any]]) -> bool:
    """Only spend an OCR call when a rule could actually fire for this step."""
    if "guide_already_open" in rules and _is_guide_press(action, arguments):
        return True
    if "destination_already_open" in rules and _is_press(action, arguments, "a"):
        return True
    return bool(matching_lessons(action, arguments, lessons))


def matching_lessons(action: str, arguments: dict[str, Any],
                     lessons: list[dict[str, Any]]) -> list[str]:
    hits = []
    for lesson in lessons or []:
        failed = lesson.get("failed_step") or {}
        if (failed.get("action") == action
                and (failed.get("arguments") or {}) == (arguments or {})):
            hits.append(str(lesson.get("id")))
    return hits


def decide(action: str, arguments: dict[str, Any], screen_text: str,
           rules: dict[str, list[str]],
           lessons: list[dict[str, Any]],
           destinations: list[str] | None = None) -> dict[str, Any] | None:
    """Return a decision dict, or None when no lesson is relevant to this step.

    decision["action"] is "skip" (do not dispatch) or "proceed" (dispatch, but
    the lesson was checked against the live screen first).

    destinations: the scenario's navigation targets (Achievements, Stats...).
    """
    if ("destination_already_open" in rules and destinations
            and _is_press(action, arguments, "a")):
        heading = _destination_heading(screen_text, destinations)
        if heading:
            return {"rule": "destination_already_open",
                    "lessons": rules["destination_already_open"], "action": "skip",
                    "reason": (f"'{heading}' is already the open page; pressing A "
                               f"would act on an item inside it.")}

    if "guide_already_open" in rules and _is_guide_press(action, arguments):
        marker = guide_is_open(screen_text)
        if marker:
            return {
                "rule": "guide_already_open",
                "lessons": rules["guide_already_open"],
                "action": "skip",
                "reason": (f"Guide overlay is already open (screen shows "
                           f"'{marker}'); pressing Guide would close it."),
            }
        return {
            "rule": "guide_already_open",
            "lessons": rules["guide_already_open"],
            "action": "proceed",
            "reason": "Guide overlay is not open; pressing Guide is safe.",
        }

    hits = matching_lessons(action, arguments, lessons)
    if hits:
        return {
            "rule": "repeat_of_failing_step",
            "lessons": hits,
            "action": "proceed",
            "reason": ("This exact step preceded a recorded failure. Proceeding "
                       "because the plan accounted for it; screen recorded for audit."),
        }
    return None
