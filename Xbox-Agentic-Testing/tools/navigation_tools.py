"""Closed-loop, text-guided navigation helpers for menu scenarios."""

from __future__ import annotations

import time
from typing import Any

from core import text_match
from core.artifacts import slug
from tools.registry import ToolContext, ToolSpec, fail, make_tool, ok
from tools.vision_tools import (
    _capture_frame,
    _invoke,
    _wait_for_stable_screen,
    detect_focus_highlight_impl,
    read_screen_text_impl,
)


def _settle(ctx: ToolContext, label: str, min_wait: float,
            timeout: float) -> dict[str, Any]:
    """Return a frame taken after the menu animation has finished.

    run-20261014-162340 showed that a fixed 0.5 s sleep captured the screen
    from the PREVIOUS press: the Guide slides between tabs and the capture
    card buffers frames, so the evidence lagged one input behind.
    """
    time.sleep(max(0.0, min_wait))
    stable = _invoke(_wait_for_stable_screen(ctx), timeout=timeout,
                     label=label, min_elapsed_seconds=0.0)
    if stable.get("ok") and stable.get("frame_path"):
        return stable
    return _invoke(_capture_frame(ctx), label=label)


def _is_page_heading(text: str, needle: str, case_sensitive: bool) -> bool:
    """Is the target the heading of the page currently shown?

    Xbox pages put their title on the first real text line (OCR'd top-down),
    e.g. 'Achievements' / 'Minecraft Dungeons II'. An exact match there means
    we are already inside that destination.
    """
    for line in text.splitlines():
        clean = line.strip()
        if sum(ch.isalpha() for ch in clean) < 3:
            continue
        if case_sensitive:
            return clean == needle
        return text_match.normalise(clean) == text_match.normalise(needle)
    return False


def _focus_matches(ctx: ToolContext, frame_path: str,
                   target: str) -> dict[str, Any]:
    """Is the target the highlighted item, not merely somewhere on screen?"""
    focus = detect_focus_highlight_impl(
        ctx, frame_path=frame_path, expected_label=target)
    # The detector falls back to a whole-frame text match when it finds no
    # green highlight. That fallback is exactly the "text is visible" signal
    # this check exists to reject, so it does not count as focus.
    has_box = focus.get("highlight_bbox") is not None
    return {
        "focused": bool(focus.get("ok") and focus.get("matched") and has_box),
        "focused_label": str(focus.get("selected_label")
                             or focus.get("observed_text") or "")[:200],
        "highlight_found": has_box or bool(focus.get("detections")),
    }


def _trace_name(ctx: ToolContext, target: str) -> str:
    """Unique per call, so a second navigation never overwrites the first."""
    count = int(ctx.scratch.get("guided_nav_count", 0)) + 1
    ctx.scratch["guided_nav_count"] = count
    return f"guided-navigation-{count:02d}-{slug(target, 30)}.json"


def _guided_navigate_to_text(ctx: ToolContext) -> Any:
    def run(target_text: str, direction: str = "right",
            max_attempts: int = 8, max_presses: int | None = None,
            interval: float = 0.8,
            case_sensitive: bool = False,
            require_focus: bool = True,
            settle_timeout: float = 4.0) -> dict[str, Any]:
        """Explore one menu direction while recording OCR until target text appears."""
        target = str(target_text).strip()
        direction_name = str(direction).strip().lower()
        if not target:
            return fail("target_text is required.")
        if direction_name not in {"left", "right", "up", "down"}:
            return fail(
                f"Unsupported direction '{direction}'. Use left, right, up or down.")

        try:
            pad = ctx.hardware.pad()
            _, canonical = ctx.hardware.controls.resolve(direction_name)
        except Exception as exc:
            return fail(f"Controller unavailable: {exc}")

        press_limit = (
            max(0, int(max_presses))
            if max_presses is not None
            else max(0, int(max_attempts) - 1)
        )
        press_limit = min(press_limit, 50)
        attempts: list[dict[str, Any]] = []
        needle = target if case_sensitive else target.lower()
        for index in range(press_limit + 1):
            label = f"guided-nav-{direction_name}-{index}"
            captured = (
                _invoke(_capture_frame(ctx), label=label) if ctx.dry_run
                else _settle(ctx, label, 0.0 if index == 0 else interval,
                             settle_timeout))
            if not captured.get("ok"):
                return fail(
                    f"Could not capture guided-navigation frame: "
                    f"{captured.get('error')}", attempts=attempts)

            reading = read_screen_text_impl(
                ctx, frame_path=captured.get("frame_path"))
            if not reading.get("ok"):
                return fail(
                    f"Could not read guided-navigation frame: "
                    f"{reading.get('error')}", attempts=attempts)

            text = str(reading.get("text", ""))
            # Exact after normalising case/punctuation/spacing: this decides
            # where input stops, so OCR letter errors are NOT tolerated here
            # (fuzzy would match "Achievement tracker" for "Achievements").
            matched_lines = (
                [line.strip() for line in text.splitlines() if target in line]
                if case_sensitive else text_match.exact_matched_lines(text, target))
            focus = (_focus_matches(ctx, str(captured.get("frame_path")), target)
                     if matched_lines and require_focus else
                     {"focused": bool(matched_lines), "focused_label": "",
                      "highlight_found": False})
            # run-20261015-113636: an earlier A press had already OPENED the
            # Achievements page. Its heading was the first OCR line, so the
            # target was the destination itself, not an item to focus - yet the
            # tool scrolled the page 7 times looking for a highlight on it.
            already_open = (index == 0 and require_focus
                            and _is_page_heading(text, needle, case_sensitive))
            reached = bool(matched_lines) and (focus["focused"] or already_open)
            observation = {
                "attempt": index + 1,
                "direction": direction_name,
                "frame_path": captured.get("frame_path"),
                "screen_text": text[:4000],
                "visible_lines": [line for line in text.splitlines()
                                  if line.strip()],
                "text_visible": bool(matched_lines),
                "already_open": already_open,
                "focused": focus["focused"],
                "focused_label": focus["focused_label"],
                "matched": reached,
                "matched_lines": matched_lines,
            }
            attempts.append(observation)
            if reached:
                ctx.artifacts.save_json(
                    _trace_name(ctx, target),
                    {"target_text": target, "direction": direction_name,
                     "found": True, "attempts": attempts})
                return ok(
                    target_text=target,
                    found=True,
                    matched_lines=matched_lines,
                    attempts=attempts,
                    attempts_used=index + 1,
                    frame_path=captured.get("frame_path"),
                    focused=focus["focused"],
                    already_open=already_open,
                    observation=(
                        f"'{target}' is " + (
                            "already the open page - do NOT press A again"
                            if already_open else
                            "the highlighted item" if require_focus else "visible")
                        + "; navigation stopped before another input was sent."),
                )

            if index >= press_limit:
                break

            if ctx.dry_run:
                observation["dispatched"] = False
                observation["dry_run"] = True
                break

            try:
                dispatched = bool(pad.press_times(canonical, 1, None, None))
            except Exception as exc:
                return fail(
                    f"Could not dispatch dpad_{direction_name}: {exc}",
                    attempts=attempts)
            observation["dispatched"] = dispatched

        seen = any(a.get("text_visible") for a in attempts)
        ctx.artifacts.save_json(
            _trace_name(ctx, target),
            {"target_text": target, "direction": direction_name,
             "found": False, "text_seen": seen, "attempts": attempts})
        reason = (
            f"'{target}' was visible but never became the highlighted item"
            if seen else f"'{target}' was not observed in captured OCR text")
        return fail(
            f"Did not reach '{target}' after {len(attempts)} guided "
            f"navigation attempt(s): {reason}.",
            target_text=target, found=False, text_seen=seen,
            attempts=attempts, attempts_used=len(attempts),
            observation=reason + ".")

    return make_tool(
        run, "guided_navigate_to_text",
        "Capture the current menu, OCR and record all visible text, move one "
        "D-pad direction, wait for the screen to settle, and repeat until "
        "target_text is the HIGHLIGHTED item (require_focus=true, default) or "
        "merely visible (require_focus=false, use for tab headings). Stops "
        "before sending another input once reached. max_presses limits "
        "directional inputs; the initial screen is always observed first.",
    )


def provide() -> list[ToolSpec]:
    return [
        ToolSpec(
            "guided_navigate_to_text",
            "Closed-loop OCR-guided menu navigation with an evidence log.",
            ["input", "vision", "analysis"],
            _guided_navigate_to_text,
            mutates_hardware=True,
        )
    ]
