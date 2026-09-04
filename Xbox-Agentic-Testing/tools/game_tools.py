"""Game navigation tools for the Xbox home screen.

The tool deliberately does not assume that the requested title is always on a
fixed tile. It checks the currently focused tile first, then the next tile, and
uses OCR evidence to decide whether the requested game is present before
pressing A. This supports the common first-tile/second-tile setup while keeping
selection evidence-based.
"""

from __future__ import annotations

import re
import time
from typing import Any

from registry import ToolContext, ToolSpec, fail, make_tool, ok
from vision_tools import read_screen_text_impl


def _pad(ctx: ToolContext) -> Any:
    return ctx.hardware.pad()


def _normalise(value: str) -> str:
    value = str(value).lower()
    value = re.sub(r"[^a-z0-9 ]+", " ", value)
    return " ".join(value.split())


def _title_matches(requested: str, observed: str) -> bool:
    want = _normalise(requested)
    text = _normalise(observed)
    if not want or not text:
        return False
    if want in text:
        return True
    wanted_words = [w for w in want.split() if len(w) > 2]
    if not wanted_words:
        return False
    return all(word in text for word in wanted_words)


def _observe(ctx: ToolContext, label: str) -> dict[str, Any]:
    frame = ctx.hardware.capture().grab(allow_blank=True)
    if frame is None:
        return fail("Capture returned no frame while locating the game.")
    path = ctx.artifacts.save_frame(frame, label)
    result = read_text_region_impl(ctx, frame_path=path)
    if not result.get("ok"):
        return fail(result.get("error", "OCR failed while locating the game."),
                    frame_path=path)
    return ok(frame_path=path, text=result.get("text", ""),
              engine=result.get("engine"))


def _discover_game(ctx: ToolContext) -> Any:
    def run(game_name: str, max_tiles: int = 2,
            move_control: str = "right") -> dict[str, Any]:
        """Find a requested game on the current or next home-screen tile."""
        requested = str(game_name).strip()
        if not requested:
            return fail("game_name is required.")
        max_tiles = max(1, min(int(max_tiles), 2))
        pad = _pad(ctx)
        observations: list[dict[str, Any]] = []

        for tile_index in range(1, max_tiles + 1):
            time.sleep(0.25)
            observed = _observe(ctx, f"game-tile-{tile_index}")
            observations.append({
                "tile": tile_index,
                "frame_path": observed.get("frame_path"),
                "text": observed.get("text", ""),
                "engine": observed.get("engine"),
                "match": _title_matches(requested, observed.get("text", "")),
            })
            if observed.get("ok") and observations[-1]["match"]:
                return ok(
                    game_name=requested,
                    found=True,
                    tile_index=tile_index,
                    frame_path=observed.get("frame_path"),
                    observed_text=observed.get("text", ""),
                    observations=observations,
                    selection_verified=True,
                )

            if tile_index < max_tiles:
                moved = pad.press(move_control)
                if not moved:
                    return fail("Could not move to the next game tile.",
                                game_name=requested, observations=observations)
                time.sleep(0.5)

        return fail(
            f"Game '{requested}' was not visually identified on the first "
            f"{max_tiles} tile(s). No game was launched.",
            game_name=requested,
            found=False,
            observations=observations,
        )

    return make_tool(
        run,
        "discover_game",
        "Find a named game on the Xbox home screen. Checks the currently "
        "focused tile first and then the next tile, using OCR before any A "
        "press. Never launches an unverified tile.",
    )


def _launch_game(ctx: ToolContext) -> Any:
    def run(game_name: str, max_tiles: int = 2,
            launch_wait: float = 8.0) -> dict[str, Any]:
        """Discover and launch a named game, then capture launch evidence."""
        discovered = _discover_game(ctx)(game_name=game_name, max_tiles=max_tiles)
        if not discovered.get("ok"):
            return discovered

        pressed = _pad(ctx).press("a")
        if not pressed:
            return fail("Game tile was identified but A was not dispatched.",
                        game_name=game_name,
                        tile_index=discovered.get("tile_index"),
                        discovery=discovered)

        # Use a bounded wait rather than treating the button acknowledgement as
        # proof. The executor/verifier will still capture and judge the result.
        time.sleep(max(0.5, min(float(launch_wait), 30.0)))
        after = _observe(ctx, "game-launch-after")
        return ok(
            game_name=game_name,
            tile_index=discovered.get("tile_index"),
            dispatched=True,
            discovery=discovered,
            launch_frame=after.get("frame_path"),
            launch_screen_text=after.get("text", ""),
            selection_verified=True,
            caveat=(
                "The game tile was identified before A was pressed. Launching "
                "is still not considered proof of successful game startup; "
                "the subsequent screen must be verified."
            ),
        )

    return make_tool(
        run,
        "launch_game",
        "Automatically locate a named game on tile 1 or tile 2, select the "
        "visually verified tile, press A, and capture launch evidence. Does "
        "not guess when the title cannot be identified.",
    )


def provide() -> list[ToolSpec]:
    return [
        ToolSpec(
            name="discover_game",
            description="Locate a requested game on the first or second home-screen tile using OCR.",
            tags=["input", "vision", "game"],
            factory=_discover_game,
            mutates_hardware=True,
        ),
        ToolSpec(
            name="launch_game",
            description="Locate a requested game on tile 1 or 2, select it, launch it, and capture evidence.",
            tags=["input", "vision", "game"],
            factory=_launch_game,
            mutates_hardware=True,
        ),
    ]
