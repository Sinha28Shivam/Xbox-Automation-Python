"""coordinate_tools.py - read the player's on-screen coordinate HUD.

Depends on the player having manually enabled Minecraft Bedrock's World
Settings -> World -> Show Coordinates (there is no F3/debug-overlay toggle on
Bedrock - that is Java Edition only).

HARDWARE-VERIFIED FORMAT (2026-09-25, real captured frame at 1920x1080)
------------------------------------------------------------------------
The HUD renders a SINGLE line, top-left, reading "Position: X, Y, Z"
comma-separated (e.g. "Position: -438, 63, 978") - NOT the "X: # Y: # Z: #"
format this tool originally guessed before a real frame was checked.
`DEFAULT_HUD_REGION` (x=0, y=0.18, width=0.30, height=0.10 of the frame) was
cropped from that real frame and visually confirmed to tightly frame the
text with margin to spare. Raw tesseract on that crop returned
"Fosition: -438, 63, 978" (P misread as F) - the numbers came through
perfectly, so the parser deliberately anchors ONLY on the three
comma-separated numbers via regex, not on the label word, making it immune
to that kind of label misread. Every result still carries a `raw_text`
field so a human/agent can catch a genuinely bad crop or misparse.

Coordinates only ever describe the PLAYER's own position - Minecraft has no
overlay or API that reveals village/structure/resource locations. This tool
is a bookkeeping primitive (letting other tools log "I was at X,Y,Z when I
saw a village"), not a discovery mechanism.
"""

from __future__ import annotations

import re
from typing import Any

from registry import ToolContext, ToolSpec, fail, make_tool, ok
from vision_tools import _capture_frame, _crop_frame, _invoke

# Matches the three comma-separated numbers in "Position: -438, 63, 978",
# deliberately NOT anchored on the word "Position" - hardware-verified OCR
# misread that label as "Fosition" while reading the digits perfectly, so
# anchoring on the numbers alone is the more robust choice.
_COORD_PATTERN = re.compile(
    r"(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)",
)

# Hardware-verified against a real 1920x1080 capture (2026-09-25): tightly
# frames the "Position: X, Y, Z" line top-left, with margin to spare.
DEFAULT_HUD_REGION = {"x": 0.0, "y": 0.18, "width": 0.30, "height": 0.10}


def read_player_coordinates_impl(
    ctx: ToolContext,
    frame_path: str | None = None,
    region: dict[str, float] | None = None,
) -> dict[str, Any]:
    """OCR the coordinate HUD and parse it into (x, y, z).

    Requires Show Coordinates to already be ON in-game - this tool only
    reads the overlay, it cannot enable the setting itself.
    """
    if not ctx.settings.get("verification.ocr.enabled", True):
        return fail("OCR is disabled in settings (verification.ocr.enabled)")

    path = frame_path or ctx.scratch.get("last_frame_path")
    if not path:
        result = _invoke(_capture_frame(ctx))
        if not result.get("ok"):
            return result
        path = result["frame_path"]

    resolved = dict(region) if region else dict(DEFAULT_HUD_REGION)
    cropped, error = _crop_frame(str(path), resolved)
    if cropped is None:
        return fail(error, frame_path=str(path), region=resolved)

    # Label with the SOURCE frame's own name, not a fixed string - a fixed
    # name was overwritten on every call, making it impossible to inspect a
    # specific cycle's crop after the fact (found while diagnosing a rain-
    # particle OCR misread that could only be root-caused by viewing the
    # exact crop that produced it).
    from pathlib import Path
    crop_label = f"crop-coordinates-hud-{Path(path).stem}"
    crop_path = ctx.artifacts.save_frame(cropped, crop_label)
    # NOT vision_tools._ocr: its multi-variant pipeline is tuned for busy,
    # low-contrast GAMEPLAY scenes, and hardware-verified (2026-09-25) to
    # sometimes pick a WORSE variant on this clean white-on-black HUD crop
    # (one real run dropped the first number entirely). A direct read is
    # simpler and was confirmed reliable across multiple real crops.
    try:
        import pytesseract
        text = pytesseract.image_to_string(cropped).strip()
        engine = "pytesseract_direct"
    except Exception as exc:
        return fail(f"No OCR engine available ({exc}).",
                    frame_path=str(path), crop_path=str(crop_path), region=resolved)
    if not text:
        return fail("OCR read no text from the HUD crop.",
                    frame_path=str(path), crop_path=str(crop_path), region=resolved)

    match = _COORD_PATTERN.search(text)
    if not match:
        return fail(
            "Coordinate HUD text was not recognized in this crop. Confirm "
            "World Settings -> World -> Show Coordinates is ON in-game, and "
            "that `region` still covers the HUD (DEFAULT_HUD_REGION was "
            "hardware-verified against one real frame, but a different "
            "aspect ratio/resolution could shift it). Check `raw_text` and "
            "`crop_path` to see what was actually read/cropped.",
            frame_path=str(path), crop_path=str(crop_path), region=resolved,
            raw_text=text, engine=engine)

    x, y, z = (float(match.group(i)) for i in (1, 2, 3))
    return ok(
        x=x, y=y, z=z,
        raw_text=text, engine=engine,
        frame_path=str(path), crop_path=str(crop_path), region=resolved,
        caveat=("Parsed from OCR text, not a game API - misreads of small "
                "HUD digits are possible. Treat a single read as a "
                "hypothesis; cross-check against a couple of real frames "
                "before trusting it for navigation."),
    )


def _read_player_coordinates(ctx: ToolContext) -> Any:
    def run(frame_path: str | None = None,
            region: dict[str, float] | None = None) -> dict[str, Any]:
        return read_player_coordinates_impl(ctx, frame_path=frame_path, region=region)

    return make_tool(
        run, "read_player_coordinates",
        "OCR the on-screen X/Y/Z coordinate HUD (Minecraft Bedrock: World "
        "Settings -> Show Coordinates must already be ON) and parse it into "
        "numeric x/y/z. Returns raw_text and crop_path alongside the parsed "
        "values so a bad crop or misread is visible, not silently trusted.")


def provide() -> list[ToolSpec]:
    return [
        ToolSpec(name="read_player_coordinates",
                 description="OCR the player's on-screen X/Y/Z coordinate HUD.",
                 tags=["vision", "analysis"],
                 factory=_read_player_coordinates, mutates_hardware=False),
    ]
