"""
vision_tools.py - perception. The only source of real proof in this framework.

Everything else can be faked by an optimistic agent; these tools look at actual
pixels coming off the capture card. That is why `verify_screen_changed` is the
single most important function in the project - it is the mechanical answer to
"did anything actually happen?"

WHAT THE HARDWARE LAYER ALREADY KNOWS
-------------------------------------
capture.py encodes measurements from this specific rig: the card needs ~1 s to
lock onto HDMI after opening, a truly flat frame (std 0) means no signal rather
than a dark scene, and frames cost ~1 ms once the device is open. We call its
functions instead of re-deriving any of that.

FRAME HANDLING
--------------
Frames are numpy arrays, which cannot go into an LLM message or a JSON tool
result. So tools save frames to the artifact store and return PATHS. The vision
agent later re-loads and base64-encodes only the frames it needs, which keeps
tool results small and gives humans the same images to inspect afterwards.
"""

from __future__ import annotations

import base64
import re
import time
import tempfile
from pathlib import Path
from typing import Any

from registry import ToolContext, ToolSpec, fail, make_tool, ok


def _capture(ctx: ToolContext) -> Any:
    return ctx.hardware.capture()


def _fns(ctx: ToolContext) -> dict[str, Any]:
    return ctx.hardware.capture_functions()


def _remember(ctx: ToolContext, frame: Any, path: str | None) -> None:
    """Keep the latest frame so a later comparison has a baseline."""
    ctx.scratch["last_frame"] = frame
    ctx.scratch["last_frame_path"] = path


def _invoke(tool: Any, **kwargs: Any) -> dict[str, Any]:
    """Call another tool from inside a tool.

    Always by KEYWORD. `make_tool` wraps every function in an
    argument-tolerant shim that accepts **kwargs only, so a positional call
    raises "takes 0 positional arguments". Routing every internal call through
    this helper keeps that detail in one place instead of scattered across the
    tools that compose others.
    """
    func = getattr(tool, "func", tool)
    return func(**kwargs)


def _text_entry_layouts(ctx: ToolContext) -> dict[str, Any]:
    controls = getattr(ctx.hardware.controls, "data", {}) or {}
    section = controls.get("text_entry", {}) or {}
    return dict(section.get("keyboard_layouts", {}) or {})


def _resolve_region(ctx: ToolContext, preset: str = "",
                    keyboard_variant: str | None = None,
                    region: dict[str, float] | None = None) -> tuple[dict[str, float] | None, str]:
    if region:
        return dict(region), "explicit"

    layouts = _text_entry_layouts(ctx)
    if preset != "search_box":
        return None, ""

    variant = keyboard_variant
    if not variant:
        defaults = ((getattr(ctx.hardware.controls, "data", {}) or {})
                    .get("text_entry", {}) or {}).get("defaults", {}) or {}
        variant = str(defaults.get("keyboard_variant", "")).strip()

    if not variant or variant not in layouts:
        return None, ""

    layout = layouts.get(variant, {}) or {}
    found = layout.get("search_box_region")
    return (dict(found), f"{variant}:search_box") if isinstance(found, dict) else (None, "")


def _crop_frame(path: str, region: dict[str, float]) -> tuple[Any | None, str]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(f"OpenCV unavailable: {exc}") from exc

    frame = cv2.imread(str(path))
    if frame is None:
        return None, f"Could not read frame: {path}"

    h, w = frame.shape[:2]
    x = float(region.get("x", 0.0))
    y = float(region.get("y", 0.0))
    rw = float(region.get("width", region.get("w", 1.0)))
    rh = float(region.get("height", region.get("h", 1.0)))

    if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 < rw <= 1.0 and 0.0 < rh <= 1.0:
        left = max(0, min(w - 1, int(round(x * w))))
        top = max(0, min(h - 1, int(round(y * h))))
        right = max(left + 1, min(w, int(round((x + rw) * w))))
        bottom = max(top + 1, min(h, int(round((y + rh) * h))))
    else:
        left = max(0, int(round(x)))
        top = max(0, int(round(y)))
        right = min(w, left + max(1, int(round(rw))))
        bottom = min(h, top + max(1, int(round(rh))))

    if left >= right or top >= bottom:
        return None, f"Region is outside the frame bounds: {region}"

    return frame[top:bottom, left:right], ""


def read_text_region_impl(ctx: ToolContext, frame_path: str | None = None,
                          region: dict[str, float] | None = None,
                          preset: str = "",
                          keyboard_variant: str | None = None) -> dict[str, Any]:
    if not ctx.settings.get("verification.ocr.enabled", True):
        return fail("OCR is disabled in settings (verification.ocr.enabled)")

    path = frame_path or ctx.scratch.get("last_frame_path")
    if not path:
        result = _invoke(_capture_frame(ctx))
        if not result.get("ok"):
            return result
        path = result["frame_path"]

    resolved, source = _resolve_region(
        ctx, preset=preset, keyboard_variant=keyboard_variant, region=region)
    if not resolved:
        return fail(
            "No text region could be resolved. Pass an explicit region or a "
            "supported preset/keyboard_variant.",
            preset=preset,
            keyboard_variant=keyboard_variant,
        )

    cropped, error = _crop_frame(str(path), resolved)
    if cropped is None:
        return fail(error, frame_path=str(path), region=resolved)

    try:
        import cv2
    except ImportError as exc:
        return fail(f"OpenCV unavailable: {exc}")

    crop_path = ctx.artifacts.save_frame(cropped, f"crop-{preset or 'region'}")
    text, engine, ocr_error = _ocr(ctx, str(crop_path))
    if text is None:
        return fail(
            f"No OCR engine available ({ocr_error}).",
            frame_path=str(path),
            crop_path=str(crop_path),
            region=resolved,
            region_source=source,
        )

    return ok(
        text=text,
        engine=engine,
        frame_path=str(path),
        crop_path=str(crop_path),
        region=resolved,
        region_source=source,
        line_count=len([l for l in text.splitlines() if l.strip()]),
        caveat=(
            "Region OCR is stronger than whole-screen OCR for search fields, "
            "but stylised UI text and low contrast can still cause misses."
        ),
    )


# ===========================================================================
# Capture
# ===========================================================================
def _capture_frame(ctx: ToolContext) -> Any:
    def run(label: str = "frame", step: int | None = None) -> dict[str, Any]:
        try:
            cam = _capture(ctx)
            fns = _fns(ctx)
        except Exception as exc:
            return fail(f"Capture unavailable: {exc}")

        # allow_blank=True on purpose: a blank frame is itself a finding worth
        # reporting (no signal / HDCP / console asleep). Hiding it as "no frame"
        # would throw away the most useful diagnostic we have.
        frame = cam.grab(allow_blank=True)
        if frame is None:
            return fail("Capture returned no frame at all - the device may "
                        "have been taken by another application.")

        stats = fns["frame_stats"](frame)
        blank = bool(fns["is_blank"](
            frame, ctx.threshold("blank_std_threshold", 1.0)))
        path = ctx.artifacts.save_frame(frame, label, step)
        _remember(ctx, frame, path)

        return ok(
            frame_path=path,
            width=int(frame.shape[1]),
            height=int(frame.shape[0]),
            blank=blank,
            stats={k: round(float(v), 3) for k, v in stats.items()},
            note=None if not blank else (
                "Frame is FLAT (std ~0) = NO HDMI SIGNAL. Not a dark scene. "
                "Check the console is awake, HDMI is in the card's IN port, "
                "and the content is not HDCP-protected."),
        )

    return make_tool(
        run, "capture_frame",
        "Grab one frame from the capture card, save it, and report its size, "
        "brightness statistics and whether it is blank. Call before and after "
        "an action so the change can be measured.")


def _wait_for_stable_screen(ctx: ToolContext) -> Any:
    def run(timeout: float | None = None,
            label: str = "stable") -> dict[str, Any]:
        try:
            cam = _capture(ctx)
        except Exception as exc:
            return fail(f"Capture unavailable: {exc}")

        started = time.time()
        frame = cam.wait_for_stable_screen(
            timeout=float(timeout or ctx.threshold("stability_timeout", 10.0)),
            settle=ctx.threshold("stability_settle", 0.4),
            threshold=ctx.threshold("screen_change_threshold", 0.5),
        )
        if frame is None:
            return fail("No frame while waiting for the screen to settle.")

        path = ctx.artifacts.save_frame(frame, label)
        _remember(ctx, frame, path)
        return ok(frame_path=path, waited_seconds=round(time.time() - started, 2))

    return make_tool(
        run, "wait_for_stable_screen",
        "Block until consecutive frames stop differing - i.e. the UI animation "
        "has finished - then return that frame. Always prefer this to a fixed "
        "sleep: it adapts to how long the console actually took.")


# ===========================================================================
# Verification - the anti-false-pass core
# ===========================================================================
def _verify_screen_changed(ctx: ToolContext) -> Any:
    def run(before_path: str, after_path: str | None = None,
            threshold: float | None = None) -> dict[str, Any]:
        try:
            import cv2
            fns = _fns(ctx)
        except Exception as exc:
            return fail(f"Vision unavailable: {exc}")

        before = cv2.imread(str(before_path))
        if before is None:
            return fail(f"Could not read the 'before' frame: {before_path}")

        if after_path:
            after = cv2.imread(str(after_path))
            if after is None:
                return fail(f"Could not read the 'after' frame: {after_path}")
        else:
            try:
                after = _capture(ctx).grab(allow_blank=True)
            except Exception as exc:
                return fail(f"Capture unavailable: {exc}")
            if after is None:
                return fail("Could not grab an 'after' frame.")
            after_path = ctx.artifacts.save_frame(after, "after-compare")

        delta = float(fns["difference"](before, after))
        limit = float(threshold if threshold is not None
                      else ctx.threshold("screen_change_threshold", 0.5))
        changed = delta >= limit

        return ok(
            changed=changed,
            delta=round(delta, 4),
            threshold=limit,
            before_path=str(before_path),
            after_path=str(after_path),
            # Spelled out because "no change" has two very different meanings
            # and confusing them sends the RCA agent down the wrong path.
            interpretation=(
                "The screen changed, so the console did receive and act on "
                "input." if changed else
                "The screen did NOT change. Either the input never reached the "
                "console (unauthenticated GIMX session is the usual cause), or "
                "it reached it and the console ignored it - a possible defect. "
                "These are different failures; do not treat them as one."),
        )

    return make_tool(
        run, "verify_screen_changed",
        "Compare two frames and report whether the picture actually changed. "
        "This is the primary proof that an action had an effect. Without it, "
        "a 'successful' button press proves nothing.")


def _compare_frames(ctx: ToolContext) -> Any:
    def run(path_a: str, path_b: str) -> dict[str, Any]:
        try:
            import cv2
            fns = _fns(ctx)
        except Exception as exc:
            return fail(f"Vision unavailable: {exc}")

        a, b = cv2.imread(str(path_a)), cv2.imread(str(path_b))
        if a is None or b is None:
            return fail("One or both frames could not be read.")

        return ok(
            delta=round(float(fns["difference"](a, b)), 4),
            stats_a={k: round(float(v), 3) for k, v in fns["frame_stats"](a).items()},
            stats_b={k: round(float(v), 3) for k, v in fns["frame_stats"](b).items()},
        )

    return make_tool(
        run, "compare_frames",
        "Numerically compare two saved frames: mean pixel difference plus "
        "brightness statistics for each.")


# ===========================================================================
# OCR
# ===========================================================================
def read_screen_text_impl(ctx: ToolContext, frame_path: str | None = None) -> dict[str, Any]:
    if not ctx.settings.get("verification.ocr.enabled", True):
        return fail("OCR is disabled in settings (verification.ocr.enabled)")

    path = frame_path or ctx.scratch.get("last_frame_path")
    if not path:
        result = _invoke(_capture_frame(ctx))
        if not result.get("ok"):
            return result
        path = result["frame_path"]

    text, engine, error = _ocr(ctx, str(path))
    if text is None:
        return fail(
            f"No OCR engine available ({error}). Install one, e.g. "
            f"pip install pytesseract, or set verification.ocr.enabled "
            f"to false and rely on frame differencing.")

    return ok(
        text=text,
        engine=engine,
        frame_path=str(path),
        line_count=len([l for l in text.splitlines() if l.strip()]),
        # Documented in docs 08: game UIs use stylised fonts over animated
        # backgrounds, so a miss is weak evidence of absence.
        caveat=(
            "OCR on console UIs is unreliable - stylised fonts, motion and "
            "transparency all hurt accuracy. Absent text is NOT strong "
            "evidence that the text is absent from the screen."),
    )


def _read_screen_text(ctx: ToolContext) -> Any:
    def run(frame_path: str | None = None) -> dict[str, Any]:
        return read_screen_text_impl(ctx, frame_path=frame_path)

    return make_tool(
        run, "read_screen_text",
        "Read on-screen text from a frame using OCR. Defaults to the most "
        "recently captured frame. Useful for locating menu items by name.")


def _find_text_on_screen(ctx: ToolContext) -> Any:
    def run(text: str, frame_path: str | None = None) -> dict[str, Any]:
        result = _invoke(_read_screen_text(ctx), frame_path=frame_path)
        if not result.get("ok"):
            return result

        haystack = str(result.get("text", "")).lower()
        needle = str(text).lower().strip()
        found = needle in haystack

        return ok(
            query=text,
            found=found,
            frame_path=result.get("frame_path"),
            engine=result.get("engine"),
            matched_lines=[
                line.strip()
                for line in str(result.get("text", "")).splitlines()
                if needle in line.lower()
            ],
            caveat=result.get("caveat"),
        )

    return make_tool(
        run, "find_text_on_screen",
        "Check whether a specific string appears on screen. Returns the "
        "matching lines. Treat a negative result as weak evidence - OCR "
        "misses stylised text regularly.")


def _check_for_text(ctx: ToolContext) -> Any:
    """Check several strings at once - the natural shape for "no error dialog".

    Added because the planner asked for exactly this tool by name, passing a
    list of error phrases. When a model reaches for a tool that does not exist,
    that is usually a signal the tool SHOULD exist: expressing "none of these
    appear" as five separate find_text_on_screen calls is clumsy and easy to
    get wrong.
    """
    def run(text_patterns: list[str], frame_path: str | None = None,
            match_type: str = "any",
            case_sensitive: bool = False) -> dict[str, Any]:
        result = _invoke(_read_screen_text(ctx), frame_path=frame_path)

        patterns = [str(p) for p in (text_patterns or [])]
        if not result.get("ok"):
            # OCR unavailable is NOT the same as "no error text present". A
            # criterion we could not evaluate must never read as satisfied.
            return fail(
                f"Could not read the screen, so these patterns could not be "
                f"checked: {result.get('error')}",
                patterns=patterns, checked=False)

        screen = str(result.get("text", ""))
        haystack = screen if case_sensitive else screen.lower()

        found: list[str] = []
        for pattern in patterns:
            needle = pattern if case_sensitive else pattern.lower()
            if needle.strip() and needle in haystack:
                found.append(pattern)

        matched = bool(found) if match_type == "any" else len(found) == len(patterns)

        return ok(
            patterns=patterns,
            match_type=match_type,
            found=found,
            matched=matched,
            checked=True,
            frame_path=result.get("frame_path"),
            engine=result.get("engine"),
            matched_lines=[
                line.strip() for line in screen.splitlines()
                if any((p if case_sensitive else p.lower())
                       in (line if case_sensitive else line.lower())
                       for p in patterns if p.strip())
            ],
            screen_text=screen[:2000],
            caveat=result.get("caveat"),
        )

    return make_tool(
        run, "check_for_text",
        "Check whether any (or all) of several strings appear on screen. Ideal "
        "for error checks: pass ['error', 'something went wrong'] with "
        "match_type='any'. Returns checked=false if OCR was unavailable - an "
        "unreadable screen is NOT evidence that the text is absent.")


def _read_text_region(ctx: ToolContext) -> Any:
    def run(frame_path: str | None = None,
            region: dict[str, float] | None = None,
            preset: str = "",
            keyboard_variant: str | None = None) -> dict[str, Any]:
        return read_text_region_impl(
            ctx,
            frame_path=frame_path,
            region=region,
            preset=preset,
            keyboard_variant=keyboard_variant,
        )

    return make_tool(
        run, "read_text_region",
        "OCR a focused part of the screen. Use preset='search_box' with a "
        "supported keyboard_variant for Xbox search-field verification.")


def _detect_focus_highlight(ctx: ToolContext) -> Any:
    """Find a green-highlighted menu item/tile and OCR the nearby label.

    RCA fix (run-20260904-132647): the previous version had no upper bound on
    contour size/shape, so a large incidentally-green region (a promo
    banner, a progress bar, a colour-shifted overlay tint) could win simply
    by being big and green - one such blob covered ~75% of the frame and was
    still ranked "best". That caused the agent to activate the Xbox
    dashboard's "My games & apps" toolbar icon instead of the intended Guide
    overlay menu entry, burning two extra replans before an accidental pass.
    """
    def run(frame_path: str | None = None,
            expected_label: str = "",
            region: dict[str, float] | None = None,
            max_area_ratio: float = 0.35) -> dict[str, Any]:
        path = frame_path or ctx.scratch.get("last_frame_path")
        if not path:
            result = _invoke(_capture_frame(ctx))
            if not result.get("ok"):
                return result
            path = result["frame_path"]

        try:
            import cv2
            import numpy as np
        except ImportError as exc:
            return fail(f"OpenCV unavailable: {exc}")

        frame = cv2.imread(str(path))
        if frame is None:
            return fail(f"Could not read frame: {path}")

        h, w = frame.shape[:2]
        if region:
            cropped, error = _crop_frame(str(path), region)
            if cropped is None:
                return fail(error, frame_path=str(path), region=region)
            frame = cropped
            h, w = frame.shape[:2]

        frame_area = float(w * h)
        # RCA fix: a single highlighted tile/menu row should never fill most
        # of the frame. Capping this well below "half the screen" stops
        # promo banners, progress bars and colour-shifted overlay
        # backgrounds from winning purely by being large.
        max_area_ratio = min(max(float(max_area_ratio), 0.01), 0.9)
        max_area = frame_area * max_area_ratio

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # RCA fix: narrowed from (35-95, 60-255, 60-255). That wide range
        # matched any greenish accent (notification dots, progress bars,
        # promo tiles) - not just the vivid, fairly saturated green Xbox
        # actually uses for its focus-highlight ring.
        lower = np.array([40, 90, 90], dtype=np.uint8)
        upper = np.array([85, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections: list[dict[str, Any]] = []
        oversized: list[dict[str, Any]] = []
        label_norm = _norm_label(expected_label)

        for idx, contour in enumerate(contours):
            x, y, bw, bh = cv2.boundingRect(contour)
            area = bw * bh
            if area < 500 or bw < 20 or bh < 12:
                continue

            # RCA fix: reject candidates spanning an implausible fraction of
            # the frame, or absurdly elongated slivers - neither looks like
            # a single highlighted menu row/tile. Recorded (not silently
            # dropped) so a failure result can explain what was rejected.
            aspect = max(bw / bh, bh / bw)
            if area > max_area or aspect > 15:
                oversized.append({
                    "bbox": {"x": int(x), "y": int(y),
                             "width": int(bw), "height": int(bh)},
                    "area": int(area),
                    "area_ratio": round(area / frame_area, 3),
                    "aspect_ratio": round(aspect, 2),
                    "reason": ("exceeds max_area_ratio" if area > max_area
                               else "aspect ratio too extreme"),
                })
                continue

            # RCA fix (run-20260904-163142): padding used to be asymmetric -
            # pad_x*3 on the right and none of that bias on the left - which
            # reliably swept the OCR crop into the *next* tile over on a
            # horizontal tile row (e.g. reading "Max" and "Skate 3" together
            # as one blob, which then wrongly "matched" whichever label the
            # caller was checking for). A real highlight ring already
            # outlines the full tile including its caption, so the crop only
            # needs a small, symmetric margin to catch anti-aliased border
            # pixels - not half a neighboring tile.
            pad_x = max(6, int(bw * 0.08))
            pad_y = max(6, int(bh * 0.08))
            left = max(0, x - pad_x)
            top = max(0, y - pad_y)
            right = min(w, x + bw + pad_x)
            bottom = min(h, y + bh + pad_y)
            crop = frame[top:bottom, left:right]
            crop_path = ctx.artifacts.save_frame(crop, f"focus-crop-{idx}")
            text, engine, _ = _ocr(ctx, str(crop_path)) if crop_path else ("", "", "")
            text = text or ""
            text_norm = _norm_label(text)
            match = bool(label_norm and label_norm in text_norm)
            green_pixels = int(cv2.countNonZero(mask[y:y + bh, x:x + bw]))
            fill_ratio = (green_pixels / area) if area else 0.0
            detections.append({
                "bbox": {"x": int(x), "y": int(y), "width": int(bw), "height": int(bh)},
                "area": int(area),
                "area_ratio": round(area / frame_area, 3),
                "crop_path": str(crop_path) if crop_path else None,
                "text": text[:500],
                "engine": engine,
                "matches_expected": match,
                "green_pixels": green_pixels,
                "fill_ratio": round(fill_ratio, 3),
            })

        if not detections:
            return fail(
                "No green-highlight region was detected.",
                frame_path=str(path),
                expected_label=expected_label,
                rejected_oversized=oversized[:5],
            )

        # RCA fix: prefer, in order - an OCR match on the expected label; a
        # ring/border shape (low fill_ratio) over a solid filled blob, since
        # real focus highlights outline a tile rather than paint it solid;
        # then more green pixels as a tie-breaker. The old scoring picked
        # the largest/most-solid-green blob first, which is exactly what let
        # a near-full-screen banner win over the real highlight.
        detections.sort(
            key=lambda d: (
                1 if d["matches_expected"] else 0,
                1 if d["fill_ratio"] < 0.55 else 0,
                d["green_pixels"],
            ),
            reverse=True,
        )
        best = detections[0]

        if expected_label and not best["matches_expected"]:
            return fail(
                "A green-highlight region was found, but it did not OCR-match the expected label.",
                frame_path=str(path),
                expected_label=expected_label,
                observed_text=best["text"],
                highlight_bbox=best["bbox"],
                crop_path=best["crop_path"],
                detections=detections[:5],
                rejected_oversized=oversized[:5],
            )

        return ok(
            frame_path=str(path),
            expected_label=expected_label,
            selected_label=best["text"],
            highlight_bbox=best["bbox"],
            crop_path=best["crop_path"],
            detections=detections[:5],
            rejected_oversized=oversized[:5],
            matched=bool(best["matches_expected"] or not expected_label),
            engine=best["engine"],
        )

    return make_tool(
        run, "detect_focus_highlight",
        "Detect a green-highlighted tile/menu item and OCR the nearby label. "
        "Use this before pressing A on menus where visible text alone does not prove focus. "
        "Rejects candidates covering more than max_area_ratio (default 0.35) of the frame "
        "or with an extreme aspect ratio, since a real focus highlight is one tile/row, "
        "never most of the screen.")


def warm_ocr_impl(ctx: ToolContext) -> dict[str, Any]:
    """Preload the configured OCR engines before time-sensitive input begins."""
    if not ctx.settings.get("verification.ocr.enabled", True):
        return fail("OCR is disabled in settings (verification.ocr.enabled)")

    if ctx.scratch.get("ocr_warmed"):
        return ok(warmed=True, already_warmed=True, engines=ctx.scratch.get("ocr_ready_engines", []))

    engines = ctx.settings.list_of("verification.ocr.engines") or ["pytesseract"]
    ready: list[str] = []
    errors: list[str] = []

    for engine in engines:
        name = str(engine).strip()
        try:
            _warm_engine(ctx, name)
            ready.append(name)
        except ImportError as exc:
            errors.append(f"{name}: not installed ({exc})")
        except Exception as exc:
            errors.append(f"{name}: {str(exc).splitlines()[0][:120]}")

    if not ready:
        return fail(
            f"No OCR engine could be preloaded ({'; '.join(errors)})",
            engines=engines,
        )

    ctx.scratch["ocr_warmed"] = True
    ctx.scratch["ocr_ready_engines"] = ready
    return ok(warmed=True, already_warmed=False, engines=ready, errors=errors)


def _ocr(ctx: ToolContext, path: str) -> tuple[str | None, str, str]:
    """Try each configured OCR engine in order. Returns (text, engine, error).

    An engine that imports but returns NOTHING is treated as a failure and the
    next engine is tried. On console UIs that matters: tesseract can read the
    dashboard's plain text well but miss stylised game titles that PaddleOCR
    picks up, and "no text found" would otherwise be reported as a successful
    read of an empty screen.
    """
    engines = ctx.settings.list_of("verification.ocr.engines") or ["pytesseract"]
    errors: list[str] = []

    for engine in engines:
        name = str(engine).strip()
        try:
            text = _run_engine(ctx, name, path)
        except ImportError as exc:
            errors.append(f"{name}: not installed ({exc})")
            continue
        except Exception as exc:
            errors.append(f"{name}: {str(exc).splitlines()[0][:120]}")
            continue

        if text is None:
            errors.append(f"{name}: unknown engine")
        elif text.strip():
            return text, name, ""
        else:
            errors.append(f"{name}: read no text")

    return None, "", "; ".join(errors)


def _run_engine(ctx: ToolContext, name: str, path: str) -> str | None:
    """Run one named OCR engine. Returns text, or None if the name is unknown."""
    if name == "pytesseract":
        import pytesseract
        from PIL import Image

        # image_to_string works, but calling get_tesseract_version first turns
        # "the binary is not installed" into a clear error here rather than a
        # confusing one deep inside the wrapper.
        pytesseract.get_tesseract_version()
        return pytesseract.image_to_string(Image.open(path))

    if name == "paddleocr":
        engine = _get_paddle_engine(ctx)
        raw = engine.predict(path) if hasattr(engine, "predict") else engine.ocr(path)
        return "\n".join(_paddle_lines(raw))

    return None


def _warm_engine(ctx: ToolContext, name: str) -> None:
    """Load one OCR engine and validate that it can be called later."""
    if name == "pytesseract":
        import pytesseract
        pytesseract.get_tesseract_version()
        return

    if name == "paddleocr":
        _get_paddle_engine(ctx)
        # Force the first real predict during warm-up so typing does not pause
        # mid-word while PaddleOCR lazily spins up sub-models.
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            from PIL import Image, ImageDraw
            image = Image.new("RGB", (320, 80), "white")
            draw = ImageDraw.Draw(image)
            draw.text((16, 20), "warmup", fill="black")
            image.save(tmp_path)
            _run_engine(ctx, name, str(tmp_path))
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
        return

    raise ValueError(f"Unknown OCR engine '{name}'")


def _get_paddle_engine(ctx: ToolContext) -> Any:
    engine = ctx.scratch.get("paddle")
    if engine is not None:
        return engine

    from paddleocr import PaddleOCR
    # PaddleOCR's constructor keywords have churned across versions -
    # `show_log` was removed, `use_angle_cls` renamed. Try the modern
    # signature, then older ones, then bare. Pinning one spelling
    # means the tool breaks on the next release.
    #
    # `enable_mkldnn=False` works around a known PaddlePaddle/PaddleX CPU
    # bug where MKL-DNN's oneDNN backend cannot convert certain PIR
    # attributes for these OCR models, failing every call with
    # "NotImplementedError: (Unimplemented) ConvertPirAttribute2Runtime-
    # Attribute not support [...]" (see PaddlePaddle/PaddleX #4970, #5131).
    # It costs some CPU inference speed, not correctness, so it is included
    # in every candidate signature rather than only as a last resort.
    for kwargs in ({"lang": "en", "use_textline_orientation": True,
                    "enable_mkldnn": False},
                   {"lang": "en", "use_angle_cls": True,
                    "enable_mkldnn": False},
                   {"lang": "en", "enable_mkldnn": False},
                   {"enable_mkldnn": False},
                   {"lang": "en", "use_textline_orientation": True},
                   {"lang": "en", "use_angle_cls": True},
                   {"lang": "en"},
                   {}):
        try:
            engine = PaddleOCR(**kwargs)
            break
        except (TypeError, ValueError):
            continue

    if engine is None:
        raise RuntimeError("PaddleOCR could not be constructed")

    ctx.scratch["paddle"] = engine           # model load is slow; reuse it
    return engine


def _paddle_lines(raw: Any) -> list[str]:
    """Extract text from PaddleOCR output, whichever shape this version uses.

    Old: [[ [box, (text, score)], ... ]]
    New: [{"rec_texts": [...], "rec_scores": [...]}]
    """
    lines: list[str] = []
    for page in (raw or []):
        if page is None:
            continue
        if isinstance(page, dict):                     # newer dict result
            lines.extend(str(t) for t in (page.get("rec_texts") or []))
            continue
        for entry in page:                             # older nested list
            try:
                if isinstance(entry, (list, tuple)) and len(entry) > 1:
                    value = entry[1]
                    lines.append(str(value[0] if isinstance(value, (list, tuple))
                                     else value))
            except (IndexError, TypeError):
                continue
    return [l for l in lines if l.strip()]


def _norm_label(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


# ===========================================================================
# Vision-model support
# ===========================================================================
def _encode_frame_for_vision(ctx: ToolContext) -> Any:
    def run(frame_path: str, max_width: int = 1280) -> dict[str, Any]:
        try:
            import cv2
        except ImportError as exc:
            return fail(f"OpenCV unavailable: {exc}")

        frame = cv2.imread(str(frame_path))
        if frame is None:
            return fail(f"Could not read frame: {frame_path}")

        # 1080p base64 is ~2 MB and dominates both latency and token cost, so
        # we downscale. Menu text stays legible well below native resolution.
        h, w = frame.shape[:2]
        if w > max_width:
            scale = max_width / float(w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

        ok_, buf = cv2.imencode(".png", frame)
        if not ok_:
            return fail("Could not encode the frame as PNG")

        return ok(
            frame_path=str(frame_path),
            media_type="image/png",
            base64=base64.b64encode(buf.tobytes()).decode("ascii"),
            width=int(frame.shape[1]),
            height=int(frame.shape[0]),
        )

    return make_tool(
        run, "encode_frame_for_vision",
        "Load a saved frame and return it base64-encoded, downscaled, ready "
        "to attach to a vision model message.")


def _list_captured_frames(ctx: ToolContext) -> Any:
    def run() -> dict[str, Any]:
        frames = ctx.artifacts.list_frames()
        return ok(frames=frames, count=len(frames),
                  run_dir=str(ctx.artifacts.run_dir))

    return make_tool(
        run, "list_captured_frames",
        "List every frame captured during this run, in order. Useful for RCA: "
        "the sequence of screenshots is the timeline of what happened.")


# ===========================================================================
# Registration
# ===========================================================================
def provide() -> list[ToolSpec]:
    return [
        ToolSpec("capture_frame", "Grab and save one frame.",
                 ["vision"], _capture_frame),
        ToolSpec("wait_for_stable_screen", "Wait until the picture settles.",
                 ["vision", "timing"], _wait_for_stable_screen),
        ToolSpec("verify_screen_changed",
                 "Did the screen actually change? Primary proof of effect.",
                 ["vision", "analysis"], _verify_screen_changed),
        ToolSpec("compare_frames", "Numerically compare two frames.",
                 ["vision", "analysis"], _compare_frames),
        ToolSpec("read_screen_text", "OCR the current or a given frame.",
                 ["vision"], _read_screen_text),
        ToolSpec("find_text_on_screen", "Search for a string on screen.",
                 ["vision", "analysis"], _find_text_on_screen),
        ToolSpec("check_for_text", "Check several strings at once (error checks).",
                 ["vision", "analysis"], _check_for_text),
        ToolSpec("read_text_region", "OCR a focused screen region.",
                 ["vision", "analysis"], _read_text_region),
        ToolSpec("detect_focus_highlight",
                 "Find the green-highlighted item and OCR its label.",
                 ["vision", "analysis"], _detect_focus_highlight),
        ToolSpec("encode_frame_for_vision", "Base64-encode a frame for an LLM.",
                 ["vision"], _encode_frame_for_vision),
        ToolSpec("list_captured_frames", "List this run's frames in order.",
                 ["vision", "analysis", "report"], _list_captured_frames),
    ]
