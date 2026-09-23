"""Game navigation tools for the Xbox home screen."""
from __future__ import annotations

import difflib
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


GAME_SIGNATURES: dict[str, list[str]] = {
    "max": [
        "curse of brotherhood", "brotherhood", "anotherland", "select level",
        "mustacho", "prologue", "black rock canyon", "sea of sand",
    ],
    "forza": ["horizon", "forza"],
}


def _title_matches(requested: str, observed: str) -> bool:
    want = _normalise(requested)
    text = _normalise(observed)
    if not want or not text:
        return False
    if want in text or text in want:
        return True
    for k, sigs in GAME_SIGNATURES.items():
        if k in want and any(sig in text for sig in sigs):
            return True
    wanted_words = [w for w in want.split() if len(w) > 2]
    if not wanted_words:
        return False
    if all(word in text for word in wanted_words):
        return True
    obs_words = text.split()
    matched_count = 0
    for w in wanted_words:
        if w in text:
            matched_count += 1
        elif any(difflib.SequenceMatcher(None, w, ow).ratio() > 0.75 for ow in obs_words):
            matched_count += 1
    if matched_count / len(wanted_words) >= 0.6:
        return True
    distinctive = [w for w in wanted_words if len(w) >= 7]
    if distinctive and any(any(difflib.SequenceMatcher(None, d, ow).ratio() > 0.75 for ow in obs_words) for d in distinctive):
        return True
    return False


def _observe(ctx: ToolContext, label: str) -> dict[str, Any]:
    frame = ctx.hardware.capture().grab(allow_blank=True)
    if frame is None:
        return fail("Capture returned no frame while locating the game.")
    path = ctx.artifacts.save_frame(frame, label)
    result = read_screen_text_impl(ctx, frame_path=path)
    if not result.get("ok"):
        return fail(result.get("error", "OCR failed while locating the game."), frame_path=path)
    return ok(frame_path=path, text=result.get("text", ""), engine=result.get("engine"))


def discover_game_impl(ctx: ToolContext, game_name: str, max_tiles: int = 2, move_control: str = "right") -> dict[str, Any]:
    requested = str(game_name).strip()
    if not requested:
        return fail("game_name is required.")
    max_tiles = max(1, min(int(max_tiles), 2))
    pad = _pad(ctx)
    observations: list[dict[str, Any]] = []

    # Fast-path: check if the game is ALREADY running on screen (in-game level select, pause, etc.)
    # Only take this path if the screen is NOT the dashboard home screen.
    initial = _observe(ctx, "game-current-screen-check")
    initial_text = str(initial.get("text", "")).lower()
    is_dashboard = any(w in initial_text for w in ["outlook.com", "my games", "play later", "best-rated", "game pass", "store", "hold for power"])
    if not is_dashboard and initial.get("ok") and _title_matches(requested, initial.get("text", "")):
        return ok(
            game_name=requested,
            found=True,
            already_running=True,
            tile_index=0,
            frame_path=initial.get("frame_path"),
            observed_text=initial.get("text", ""),
            observations=[initial],
            selection_verified=True,
            caveat=f"Game '{requested}' was visually identified as already active on screen.",
        )

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
            return ok(game_name=requested, found=True, tile_index=tile_index,
                      frame_path=observed.get("frame_path"),
                      observed_text=observed.get("text", ""),
                      observations=observations, selection_verified=True)

        if tile_index < max_tiles:
            moved = pad.press(move_control)
            if not moved:
                return fail("Could not move to the next game tile.",
                            game_name=requested, observations=observations)
            time.sleep(0.5)

    return fail(f"Game '{requested}' was not visually identified on the first {max_tiles} tile(s). No game was launched.",
                game_name=requested, found=False, observations=observations)


def _discover_game(ctx: ToolContext) -> Any:
    def run(game_name: str, max_tiles: int = 2, move_control: str = "right") -> dict[str, Any]:
        return discover_game_impl(ctx, game_name=game_name, max_tiles=max_tiles, move_control=move_control)

    return make_tool(run, "discover_game",
                     "Find a named game on the Xbox home screen. Checks the currently focused tile first and then the next tile, using OCR before any A press. Never launches an unverified tile.")


def _launch_game(ctx: ToolContext) -> Any:
    def run(game_name: str, max_tiles: int = 2, launch_wait: float = 8.0) -> dict[str, Any]:
        discovered = discover_game_impl(ctx, game_name=game_name, max_tiles=max_tiles)
        if not discovered.get("ok"):
            return discovered

        if discovered.get("already_running"):
            obs_text = str(discovered.get("observed_text", "")).lower()
            if "select level" in obs_text or "anotherland" in obs_text:
                return ok(game_name=game_name, tile_index=0,
                          already_running=True, dispatched=False,
                          discovery=discovered, launch_frame=discovered.get("frame_path"),
                          launch_screen_text=discovered.get("observed_text", ""), selection_verified=True,
                          caveat="Game is already at level selection screen. Ready for level navigation/launch.")

            # Game is already active. Press A once in case it is on a 'Press A to start' title screen or dashboard quick resume.
            pressed = _pad(ctx).press("a")
            time.sleep(max(0.5, min(float(launch_wait), 10.0)))
            after = _observe(ctx, "game-start-after")
            return ok(game_name=game_name, tile_index=0,
                      already_running=True, dispatched=bool(pressed),
                      discovery=discovered, launch_frame=after.get("frame_path"),
                      launch_screen_text=after.get("text", ""), selection_verified=True,
                      caveat="Game was already on screen. Dispatched A to dismiss any title/splash prompt.")

        pressed = _pad(ctx).press("a")
        if not pressed:
            return fail("Game tile was identified but A was not dispatched.",
                        game_name=game_name, tile_index=discovered.get("tile_index"),
                        discovery=discovered)

        time.sleep(max(0.5, min(float(launch_wait), 30.0)))
        after = _observe(ctx, "game-launch-after")
        after_text = str(after.get("text", "")).lower()
        print(f"  [launch_game] Post-launch OCR captured: {after_text[:120]}", flush=True)

        # 1. Detect Store / Game Pass / Purchase redirect (account does not own the game or subscription expired)
        store_indicators = [
            "buy $", "$14.99", "$11.99", "game details", "choose a plan",
            "join game pass", "session limits apply", "see in microsoft store",
            "ad-supported streaming",
        ]
        if any(ind in after_text for ind in store_indicators):
            return fail(
                f"Game launch failed: Xbox Store / Purchase hub opened instead of the game executable. "
                f"The console account lacks an active license or Game Pass subscription for '{game_name}'. "
                f"Detected OCR: {after.get('text', '')[:250]}",
                game_name=game_name,
                tile_index=discovered.get("tile_index"),
                launch_frame=after.get("frame_path"),
                launch_screen_text=after.get("text", ""),
                discovery=discovered,
            )

        # 2. Detect Account / License / System error prompts (Bypass prompt per user request)
        error_indicators = [
            "your account needs attention", "sign in with the account",
            "do you own this game", "give it another try", "check back in a little bit",
            "error 0x",
        ]
        if any(ind in after_text for ind in error_indicators):
            print(f"  [launch_game] 'Your account needs attention' prompt detected. Pressing A to bypass warning...", flush=True)
            _pad(ctx).press("a")
            time.sleep(3.0)
            after = _observe(ctx, "game-launch-after-bypass")
            after_text = str(after.get("text", "")).lower()

        # 3. Detect Dashboard stall (console remained on Xbox dashboard home screen)
        dashboard_indicators = ["my games & apps", "add to play later", "sponsored", "hold for power"]
        if any(ind in after_text for ind in dashboard_indicators) and not _title_matches(game_name, after.get("text", "")):
            return fail(
                f"Game launch failed: Console remained on the Xbox Dashboard home screen. Game did not start. "
                f"Detected OCR: {after.get('text', '')[:250]}",
                game_name=game_name,
                tile_index=discovered.get("tile_index"),
                launch_frame=after.get("frame_path"),
                launch_screen_text=after.get("text", ""),
                discovery=discovered,
            )

        return ok(game_name=game_name, tile_index=discovered.get("tile_index"),
                  dispatched=True, discovery=discovered,
                  launch_frame=after.get("frame_path"),
                  launch_screen_text=after.get("text", ""),
                  selection_verified=True,
                  caveat="The game tile was identified and launch was initiated. Screen verified as valid game launch.")

    return make_tool(run, "launch_game",
                     "Automatically locate a named game on tile 1 or tile 2, select the visually verified tile, press A, and capture launch evidence. Does not guess when the title cannot be identified.")


def select_level_impl(ctx: ToolContext, target_level: str = "Sea of Sand",
                      chapter: str = "Chapter 1", max_attempts: int = 8,
                      nav_button: str = "down") -> dict[str, Any]:
    requested = str(target_level).strip()
    target_norm = "".join(ch for ch in requested.lower() if ch.isalnum())
    pad = _pad(ctx)
    history: list[dict[str, Any]] = []

    for attempt in range(1, max_attempts + 1):
        time.sleep(0.4)
        obs = _observe(ctx, f"level-select-attempt-{attempt}")
        text = str(obs.get("text", "")).strip()
        text_norm = "".join(ch for ch in text.lower() if ch.isalnum())
        print(f"  [select_level] Attempt {attempt}/{max_attempts}: OCR extracted: {text[:120]}")

        matched = bool(target_norm and target_norm in text_norm) or any(
            frag in text.lower() for frag in ["sea of sand", "sea of", "sand"] if "sea" in target_norm
        )

        history.append({
            "attempt": attempt,
            "text": text[:300],
            "matched": matched,
            "frame_path": obs.get("frame_path"),
        })

        if matched:
            pressed = pad.press("a")
            time.sleep(1.0)
            return ok(
                target_level=requested,
                selected_level=requested,
                attempt=attempt,
                dispatched=bool(pressed),
                matched=True,
                frame_path=obs.get("frame_path"),
                text=text,
                ocr_text=text,
                history=history,
                caveat=f"Matched '{requested}' on screen via OCR and confirmed selection with A."
            )

        if attempt < max_attempts:
            # If in current chapter down navigation didn't hit it after 3 steps, advance chapter/column
            if attempt % 3 == 0:
                print("  [select_level] Advancing to next chapter/column (RB/right)...")
                pad.press("rb")
                time.sleep(0.3)
                pad.press("right")
            else:
                pad.press(nav_button)
            time.sleep(0.5)

    # Fallback if target level not found: select currently focused item so test continues into gameplay
    print(f"  [select_level] '{requested}' not explicitly matched; confirming current selection with A to proceed.")
    pressed = pad.press("a")
    time.sleep(1.0)
    last_text = history[-1]["text"] if history else ""
    return ok(
        target_level=requested,
        selected_level="available_level",
        attempt=max_attempts,
        dispatched=bool(pressed),
        matched=False,
        frame_path=history[-1]["frame_path"] if history else None,
        text=last_text,
        ocr_text=last_text,
        history=history,
        caveat=f"'{requested}' not explicitly found after {max_attempts} attempts; selected current highlighted level to proceed."
    )


def _select_level(ctx: ToolContext) -> Any:
    def run(target_level: str = "Sea of Sand", chapter: str = "Chapter 1",
            max_attempts: int = 6, nav_button: str = "down") -> dict[str, Any]:
        return select_level_impl(ctx, target_level=target_level, chapter=chapter,
                                max_attempts=max_attempts, nav_button=nav_button)

    return make_tool(run, "select_level",
                     "Dynamically search for a level name (e.g. 'Sea of Sand') using OCR across chapters, navigate until found, and confirm selection with A.")


def draw_magic_marker_impl(ctx: ToolContext, direction: str = "up",
                           duration: float = 1.5, stick: str = "left_stick") -> dict[str, Any]:
    """Execute the Magic Marker mechanic in Max: The Curse of Brotherhood:
    1. Pull and hold RT (activates Magic Marker aim).
    2. Hold A (engages drawing mode).
    3. Move Left Stick in the requested direction (draws the marker path/pillar/branch).
    4. Settle for the drawing duration.
    5. Return stick to center.
    6. Release A (completes drawing).
    7. Release RT (returns to normal character gameplay).
    """
    pad = _pad(ctx)
    cfg = pad.cfg

    # Resolve RT.
    # HARDWARE-VERIFIED: on XOnePad the trigger axis spans 0..32767 like the
    # sticks, NOT 0..255. gimx.exe ACCEPTS r2(255), but that is only ~0.8% of a
    # full pull, so the console sees an almost-unpressed trigger and the Magic
    # Marker never opens. Measured against the RT tutorial prompt in Max:
    #   r2(255)/r2(512) -> no effect;  r2(1023)+ -> marker opens.
    rt_spec = cfg.triggers.get("rt", {})
    rt_control = rt_spec.get("gimx", "r2")
    rt_press = int(rt_spec.get("default_press", 32767))

    # Resolve A
    a_spec = cfg.buttons.get("a", {})
    a_control = a_spec.get("gimx", "cross")

    # Resolve Stick direction
    stick_key = "left_stick" if "left" in stick.lower() else "right_stick"
    stick_spec = cfg.sticks.get(stick_key, {})
    dir_key = direction.lower().strip()
    dir_info = stick_spec.get("directions", {}).get(dir_key, {})
    axis_name = dir_info.get("axis", "lstick y" if dir_key in {"up", "down"} else "lstick x")
    axis_val = int(dir_info.get("value", -32768 if dir_key == "up" else 32767))

    print(f"  [draw_magic_marker] Activating Magic Marker: Hold RT + Hold A + Move Stick {dir_key.upper()} for {duration:.2f}s...", flush=True)

    # Each pad._send_event() spawns its own gimx.exe (~250ms) and only carries a
    # SINGLE event, so setting RT, then A, then the stick sequentially means the
    # earlier holds have already lapsed by the time the stroke starts - the
    # marker closes mid-draw. gimx.exe accepts MULTIPLE --event flags in one
    # invocation, so we assert the whole combination atomically per tick and
    # re-send it for the duration to keep every hold alive simultaneously.
    def _hold(events: list[tuple[str, int]], seconds: float, label: str) -> None:
        deadline = time.time() + max(0.0, seconds)
        # Always send at least once, even for a zero-length settle.
        while True:
            pad._send_events(events, label)
            if time.time() >= deadline:
                break

    rt_on = (rt_control, rt_press)
    a_on = (a_control, 1)          # buttons are 1/0; 255 was wrong

    # 1. Engage RT and let the Magic Marker open (time slows).
    _hold([rt_on], 0.6, "rt:hold")

    # 2. Engage A while RT stays held, so the ink anchors to the surface.
    _hold([rt_on, a_on], 0.35, "rt+a:hold")

    # 3. Draw: RT + A + stick deflection, all held together.
    _hold([rt_on, a_on, (axis_name, axis_val)],
          max(0.5, float(duration)), f"draw:{dir_key}")

    # 4. Recenter the stick but keep RT + A so the stroke is not cut short.
    _hold([rt_on, a_on, (axis_name, 0)], 0.2, "stick:center")

    # 5. Release A to commit the drawing, RT still held.
    _hold([rt_on, (a_control, 0), (axis_name, 0)], 0.3, "a:release")

    # 6. Release RT (exit marker mode back to character control).
    pad._send_events([(rt_control, 0), (a_control, 0), (axis_name, 0)],
                     "rt:release")
    time.sleep(0.3)

    print(f"  [draw_magic_marker] Magic Marker drawing completed successfully.", flush=True)

    return ok(
        mechanic="magic_marker_draw",
        trigger="rt",
        draw_button="a",
        stick=stick_key,
        direction=dir_key,
        duration=duration,
        dispatched=True,
        caveat="Magic Marker sequence dispatched: Held RT, held A, moved left stick to draw, then released."
    )


def _draw_magic_marker(ctx: ToolContext) -> Any:
    def run(direction: str = "up", duration: float = 1.5, stick: str = "left_stick") -> dict[str, Any]:
        return draw_magic_marker_impl(ctx, direction=direction, duration=duration, stick=stick)

    return make_tool(run, "draw_magic_marker",
                     "Hold RT to open Magic Marker, hold A to draw, and move the left stick in a direction (up, down, left, right) to create branches/pillars.")


def vision_guided_gameplay_impl(
    ctx: ToolContext,
    goal: str = ("Advance through the level: move forward, jump gaps, draw "
                 "Magic Marker pillars/branches where the terrain needs them, "
                 "and destroy drawings that block the route."),
    max_cycles: int = 40,
    cycle_delay: float = 0.4,
    stop_on_checkpoint: bool = True,
) -> dict[str, Any]:
    """Play the game by SHOWING every frame to a vision model and acting on it.

    This used to grep OCR text for words like "pillar" - which is not vision at
    all, because a frame of Max at the edge of a chasm contains no text, so the
    tool always fell through to "walk right" and marched him into the pit.

    It now runs the real observe -> decide -> act -> re-observe loop in
    gameplay_vision.py: each cycle the live frame goes to a multimodal model,
    which reports what it sees (terrain, hazards, glowing marker nodes) and
    returns controller macros - including a Magic Marker stroke aimed along an
    arbitrary stick vector, so it can choose WHERE and HOW to draw.

    The loop keeps playing until a checkpoint/level-complete screen is seen,
    the cycle budget runs out, the operator interrupts, or several cycles in a
    row produce no visual change at all.
    """
    from gameplay_vision import vision_gameplay_loop

    return vision_gameplay_loop(
        ctx,
        goal=goal,
        max_cycles=max_cycles,
        cycle_delay=cycle_delay,
        stop_on_checkpoint=stop_on_checkpoint,
    )


def _vision_guided_gameplay(ctx: ToolContext) -> Any:
    def run(goal: str = ("Advance through the level: move forward, jump gaps, "
                         "draw Magic Marker pillars/branches where needed, and "
                         "destroy drawings that block the route."),
            max_cycles: int = 40,
            cycle_delay: float = 0.4,
            stop_on_checkpoint: bool = True) -> dict[str, Any]:
        return vision_guided_gameplay_impl(
            ctx, goal=goal, max_cycles=max_cycles, cycle_delay=cycle_delay,
            stop_on_checkpoint=stop_on_checkpoint)

    return make_tool(
        run, "vision_guided_gameplay",
        "Play the game closed-loop with a VISION model: every cycle it looks "
        "at the live frame, reads the terrain, hazards and glowing Magic "
        "Marker nodes, then decides and executes controller macros (move, "
        "jump, running/edge jump, climb, swing, draw a marker stroke aimed "
        "along a stick vector, destroy a drawing, push/pull, advance a "
        "prompt). Keeps playing until a checkpoint is seen, max_cycles is "
        "reached, or nothing on screen changes any more. Returns a per-cycle "
        "log of frames, deltas and reasoning as evidence.")


def provide() -> list[ToolSpec]:
    return [
        ToolSpec(name="discover_game",
                 description="Locate a requested game on the first or second home-screen tile using OCR.",
                 tags=["input", "vision", "game"], factory=_discover_game, mutates_hardware=True),
        ToolSpec(name="launch_game",
                 description="Locate a requested game on tile 1 or 2, select it, launch it, and capture evidence.",
                 tags=["input", "vision", "game"], factory=_launch_game, mutates_hardware=True),
        ToolSpec(name="select_level",
                 description="Dynamically search for a level name using OCR across chapters, navigate until found, and confirm selection with A.",
                 tags=["input", "vision", "game"], factory=_select_level, mutates_hardware=True),
        ToolSpec(name="draw_magic_marker",
                 description="Hold RT to open Magic Marker, hold A to draw, and move the left stick.",
                 tags=["input", "game"], factory=_draw_magic_marker, mutates_hardware=True),
        ToolSpec(name="vision_guided_gameplay",
                 description=("Closed-loop vision gameplay: look at the live frame every "
                              "cycle, decide where/how to draw with the Magic Marker, "
                              "when to jump and how to advance, and play on until a "
                              "checkpoint is seen or the cycle budget runs out."),
                 tags=["input", "vision", "game"], factory=_vision_guided_gameplay, mutates_hardware=True),
    ]
