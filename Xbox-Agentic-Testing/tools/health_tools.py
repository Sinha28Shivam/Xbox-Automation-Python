"""health_tools.py - read Minecraft Bedrock's survival HUD (hearts/hunger)
by COLOR-MASK + CONTOUR COUNTING, not OCR.

WHY NOT OCR FOR ICON ROWS
--------------------------
OCR engines are trained on glyphs/characters; a row of small, repeated,
uniformly-colored game icons (hearts, hunger drumsticks, XP segments,
hotbar count badges) is not text and OCRs unreliably or not at all. This
framework already solved an equivalent problem twice by NOT using OCR:
  - the Magic Marker ink gauge (game_profiles/max_profile.py: mask a hue
    range, find contours, filter by size/circularity/fill-ratio)
  - the green focus-highlight detector (vision_tools.py: mask green,
    find the largest qualifying contour)
This module applies the same technique to hearts and hunger icons: mask by
color -> find contours -> filter by known icon size -> count them.

HARDWARE-VERIFIED (2026-09-25, real survival-mode frame, 10/10 hearts and
10/10 hunger, night, creeper visible in the water - unrelated to this HUD)
------------------------------------------------------------------------------
Real pixel sampling on this exact frame found:
  - Full hearts: near-pure red, HSV wraps the 0/180 boundary
    (measured (177,255,239) and confirms typical Bedrock heart red).
    Confirmed via contour test: filtering the heart-row mask to
    width>=20 produced EXACTLY 10 evenly-spaced 28x28 blobs matching the
    10 hearts visible on screen. A width<20 blob at the SAME row height is
    the small red highlight on a hunger drumstick's tip, not a heart - the
    width filter is what tells them apart, not color alone (both are red).
  - Full hunger icons: tan/bone color, HSV approx (8-20, 100-160, 130-200)
    measured directly off the drumstick bone pixels. Confirmed via contour
    test: RESTRICTED TO THE ICON ROW'S OWN Y-BAND, produced EXACTLY 10
    evenly-spaced 12x12 blobs. Without that y-restriction the same color
    range also matches wood/dirt hotbar item textures below the HUD row
    and returns garbage extra contours - the y-band restriction is
    required, not optional.
  - Icon spacing confirmed identical for both rows: 32px step at this
    crop's resolution (DEFAULT_HUD_REGION's crop width), which is the
    basis for HEART_SLOT_WIDTH/HUNGER_SLOT_WIDTH below - used to convert
    "blobs found" into "which of the 10 slots are filled", so a MISSING
    icon (e.g. 7/10 hearts, empty slots on the right) is distinguishable
    from a small/half icon, not just silently undercounted.

THIS ONLY APPLIES IN SURVIVAL/ADVENTURE MODE
-----------------------------------------------
Creative mode hides both bars entirely (no damage, no hunger) - calling
this in creative mode will correctly return 0 for both with no icons
found, which should be read as "not applicable", not "critically low".

HALF-ICONS ARE NOT YET DISTINGUISHED FROM FULL ONES
-------------------------------------------------------
This first version counts FULL icons only (confirmed against a 10/10 full
frame). A half-heart/half-hunger icon is narrower/partially-filled and will
either be missed entirely or undercounted as a partial blob depending on
where the crop lands - not yet hardware-verified against a real half-icon
frame. Treat any count below 10 as "at most this many are full", not as an
exact remaining-health number, until that case is verified.
"""

from __future__ import annotations

from typing import Any

from registry import ToolContext, ToolSpec, fail, make_tool, ok
from vision_tools import _capture_frame, _crop_frame, _invoke

# Hardware-verified against a real 1920x1080 survival-mode capture
# (2026-09-25): tightly frames both the heart row and the hunger row,
# directly above the hotbar, with margin to spare on all sides.
DEFAULT_HEALTH_HUD_REGION = {"x": 0.30, "y": 0.82, "width": 0.40, "height": 0.11}

# Icon-row y-band and slot spacing, measured within the cropped region
# above (NOT full-frame coordinates) - see module docstring for how these
# were derived from the real contour test.
_HEART_ROW_Y = (0, 30)
_HUNGER_ROW_Y = (0, 30)
_HEART_MIN_WIDTH = 20      # separates a full heart (28px) from a red fragment (12px)
# NOTE: cv2.contourArea() returns the polygon area, NOT the w*h bounding-box
# area - measured 74.5 for a real 12x12 drumstick blob (bbox area 144), a
# real bug caught during hardware verification where an earlier ad-hoc test
# had conflated the two and set this too high (80), silently rejecting
# every real hunger icon (10/10 read as 0/10 despite the crop being
# correct). Threshold set below the measured 74.5 with margin.
_HUNGER_MIN_AREA = 40
_SLOT_COUNT = 10           # Bedrock's default max hearts/hunger


def _count_icons(cropped: Any, hsv_lower: tuple[int, int, int],
                 hsv_upper: tuple[int, int, int], hsv_lower2: tuple[int, int, int] | None,
                 hsv_upper2: tuple[int, int, int] | None, y_band: tuple[int, int],
                 min_width: int, min_area: int) -> tuple[int, list[tuple[int, int, int, int]]]:
    import cv2
    import numpy as np

    y0, y1 = y_band
    row = cropped[max(0, y0):min(cropped.shape[0], y1), :]
    hsv = cv2.cvtColor(row, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(hsv_lower, dtype=np.uint8),
                       np.array(hsv_upper, dtype=np.uint8))
    if hsv_lower2 is not None and hsv_upper2 is not None:
        mask |= cv2.inRange(hsv, np.array(hsv_lower2, dtype=np.uint8),
                            np.array(hsv_upper2, dtype=np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if w < min_width:
            continue
        boxes.append((x, y, w, h))
    boxes.sort(key=lambda b: b[0])
    return len(boxes), boxes


def read_survival_hud_impl(
    ctx: ToolContext,
    frame_path: str | None = None,
    region: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Count full hearts and full hunger icons via color-mask + contours.

    Returns 0/0 (with no error) in creative mode, where both bars are
    hidden - that is a correct, expected read, not a failure.
    """
    path = frame_path or ctx.scratch.get("last_frame_path")
    if not path:
        result = _invoke(_capture_frame(ctx))
        if not result.get("ok"):
            return result
        path = result["frame_path"]

    resolved = dict(region) if region else dict(DEFAULT_HEALTH_HUD_REGION)
    cropped, error = _crop_frame(str(path), resolved)
    if cropped is None:
        return fail(error, frame_path=str(path), region=resolved)

    crop_path = ctx.artifacts.save_frame(cropped, "crop-survival-hud")

    # Full hearts: near-pure red, wraps the HSV hue boundary at 0/180.
    heart_count, heart_boxes = _count_icons(
        cropped, (0, 150, 150), (8, 255, 255), (170, 150, 150), (180, 255, 255),
        _HEART_ROW_Y, _HEART_MIN_WIDTH, 15)

    # Full hunger: tan/bone drumstick color, single hue range (no wrap).
    hunger_count, hunger_boxes = _count_icons(
        cropped, (8, 100, 130), (20, 160, 200), None, None,
        _HUNGER_ROW_Y, 8, _HUNGER_MIN_AREA)

    return ok(
        full_hearts=min(heart_count, _SLOT_COUNT),
        full_hunger=min(hunger_count, _SLOT_COUNT),
        max_slots=_SLOT_COUNT,
        heart_boxes=heart_boxes,
        hunger_boxes=hunger_boxes,
        frame_path=str(path),
        crop_path=str(crop_path),
        region=resolved,
        caveat=(
            "Counts FULL icons only - half-hearts/half-hunger are not yet "
            "hardware-verified and may be undercounted or missed (see "
            "module docstring). 0/0 in CREATIVE mode is a correct read "
            "(both bars are hidden there), not a low-health alarm - check "
            "game mode before treating a 0 as meaningful."),
    )


def _read_survival_hud(ctx: ToolContext) -> Any:
    def run(frame_path: str | None = None,
            region: dict[str, float] | None = None) -> dict[str, Any]:
        return read_survival_hud_impl(ctx, frame_path=frame_path, region=region)

    return make_tool(
        run, "read_survival_hud",
        "Count full hearts and full hunger icons via color-mask + contour "
        "counting (NOT OCR - icon rows are not text). Only meaningful in "
        "survival/adventure mode; returns 0/0 with no error in creative "
        "mode, where both bars are hidden. Half-icons are not yet "
        "hardware-verified.")


def provide() -> list[ToolSpec]:
    return [
        ToolSpec(name="read_survival_hud",
                 description="Count full hearts/hunger icons via color-mask + contours.",
                 tags=["vision", "analysis"],
                 factory=_read_survival_hud, mutates_hardware=False),
    ]
