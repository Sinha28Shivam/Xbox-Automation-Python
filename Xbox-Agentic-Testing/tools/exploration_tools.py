"""exploration_tools.py - closed-loop outward exploration for Minecraft.

Reuses MinecraftFrameDecision/MinecraftMove and execute_minecraft_move from
game_profiles/minecraft_profile.py (the schema and dispatch are genuinely
game-specific and already exist - no reason to duplicate them for a
different goal) with its OWN system prompt, since chopping-tree-craft-
planks framing would confuse a model that is supposed to be exploring
instead.

WHY COORDINATE-DELTA STUCK DETECTION, NOT PIXEL DELTA
-------------------------------------------------------
The existing gameplay loops (gameplay_engine.run_gameplay_loop) detect being
stuck from pixel deltas between frames. That is noisy for movement across
open terrain: a camera pointed at a large uniform surface (sky, plain
grass) can register a small pixel delta even while the player is walking
normally, and a genuinely blocked walk into a wall can still show a
non-trivial delta from lighting/particle noise. Coordinate deltas (via
coordinate_tools.read_player_coordinates) measure the thing that actually
matters for exploration - did the player's world position change - and are
immune to that class of false read. Falls back to pixel delta only if the
coordinate HUD cannot be read (e.g. Show Coordinates was turned off),
so the tool still degrades to something useful rather than failing outright.

BOUNDED, NOT INDEFINITE
------------------------
Every existing gameplay loop in this framework (Max's and Minecraft's) stops
on a cycle budget plus several independent real conditions (stuck-streak,
capture/model failure, KeyboardInterrupt) - never runs open-ended. This tool
follows the same proven shape rather than wandering forever, since an
unbounded loop with no operator watching is exactly the scenario that
burned a terminal needing a manual kill earlier this session (the RCA
retry-routing bug).

THE HONESTY RULE STILL APPLIES
-------------------------------
Nothing here returns `success`. Each cycle records the before/after
coordinate reads (or frame paths if coordinates were unavailable) and the
measured distance moved; the caller decides what that evidence proves.
"""

from __future__ import annotations

import math
import time
from typing import Any

from registry import ToolContext, ToolSpec, fail, make_tool, ok

from gameplay_engine import build_vision_decider, decide, encode_frame, format_history, frame_delta
from game_profiles.minecraft_profile import MinecraftFrameDecision, MinecraftMove, execute_minecraft_moves
from coordinate_tools import read_player_coordinates_impl
from location_memory import record_location_impl
from health_tools import read_survival_hud_impl

_MOVE_DISTANCE_FLOOR = 0.5  # blocks; below this a movement cycle counts as "no progress"

# Real hardware run (2026-09-25, night): this loop had NO health awareness
# at all and the player died at cycle 3 ("YOU DIED! ... slain by Zombie") -
# the frame right before death showed health had already crashed to ~1.5
# hearts while the loop kept exploring, oblivious. combat_tools.py's
# equivalent fix (real hearts-drop retreat, hard stop at a critical floor)
# is applied here too - exploration is no safer from an unseen mob than
# combat is.
_CRITICAL_HEARTS = 4

# Hardware-confirmed real bug (2026-09-25): rain particles crossing the HUD
# digits caused OCR to drop a minus sign AND a digit ("-429" read as "29"),
# producing a fabricated ~463-block jump. Confirmed by viewing the actual
# frame: the visible on-screen text was "Position: -429, 64, 987" the whole
# time - the player never moved between reads, only the OCR read changed.
# This is checked against a PERSISTENT last-known-good anchor (not just
# before-vs-after within one cycle), because the bad read was internally
# CONSISTENT within its own cycle (both before/after reads misread the same
# way) - only a cross-cycle comparison against the last trusted value could
# catch it. A single move cycle (duration <= 6s, walking speed) cannot
# plausibly cover more than this many blocks.
_MAX_PLAUSIBLE_CYCLE_DISTANCE = 40.0


def _read_coords_checked(ctx: ToolContext, frame_path: str | None,
                        last_good: dict[str, float] | None) -> tuple[dict[str, Any], bool]:
    """Read coordinates and reject them if implausibly far from the last
    trusted read. Returns (result, was_rejected). A rejected result still
    has ok=True with the raw OCR values preserved (for debugging) but
    callers must check `was_rejected` before trusting x/y/z.
    """
    result = read_player_coordinates_impl(ctx, frame_path=frame_path)
    if not result.get("ok") or last_good is None:
        return result, False
    jump = _distance(last_good["x"], last_good["y"], last_good["z"],
                     result["x"], result["y"], result["z"])
    if jump > _MAX_PLAUSIBLE_CYCLE_DISTANCE:
        print(f"  !! Rejecting implausible coordinate read ({jump:.1f} blocks "
              f"from last trusted position) - likely OCR misread, not real "
              f"movement. raw_text={result.get('raw_text')!r}", flush=True)
        return result, True
    return result, False


EXPLORATION_SYSTEM_PROMPT = """\
You are exploring Minecraft (Xbox Bedrock Edition) on a real Xbox One, in
survival first-person view. You control it through an emulated controller.
The attached image is the LIVE screen right now - reason only from what you
can actually see in it.

GOAL: walk outward and explore the surrounding terrain, avoiding hazards
(lava, deep water, cliffs, hostile mobs). Do NOT try to mine, craft, or
fight - just move, look around to survey new terrain, and report anything
notable in `nearby_entities` (villages, structures, mobs, resources). If a
hostile mob is close, turn away and move in a different direction rather
than engaging it.

CONTROLS AVAILABLE TO YOU
  move  : left stick, walk in `direction` (forward=up, back=down)
  look  : right stick, turn the camera in `direction` to survey new terrain
  wait  : do nothing this cycle, for a screen that is still loading

Prefer alternating a few `move` cycles with an occasional `look` to survey
before continuing, rather than only ever moving in one fixed direction.
If the last few cycles show the player's position barely changing, you are
probably blocked by terrain - turn and move a different direction rather
than repeating the same move.

SCENE STATE RULES
  in_gameplay        : normal first-person view, hotbar/crosshair visible
  menu_or_prompt      : any overlay, dialog, or prompt is on screen
  cutscene_or_loading : a loading/black/transition screen with no player HUD
  stuck_or_blocked    : repeated attempts are not changing position at all
  (inventory_open / planks_crafted are not relevant to this task, but the
  schema still requires classifying honestly if either happens to appear)

Return your reading of the frame and 1-3 moves for this cycle.
"""


def _distance(x1: float, y1: float, z1: float, x2: float, y2: float, z2: float) -> float:
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)


def explore_and_map_impl(
    ctx: ToolContext,
    max_cycles: int = 30,
    # See engage_single_mob_impl's note: combined move+look dispatch removed
    # most of the sequential subprocess overhead this delay used to cover.
    cycle_delay: float = 0.2,
    settle_after_move: float = 0.45,
    stuck_limit: int = 6,
    auto_log_sightings: bool = True,
) -> dict[str, Any]:
    """Explore outward, using coordinate deltas (not pixel deltas) to detect
    being stuck. Optionally auto-records notable sightings via
    location_memory.record_location as they are observed.

    Runs until one of these ends it (bounded, like every other gameplay
    loop in this framework):
      * `max_cycles` decision cycles have run
      * `stuck_limit` consecutive cycles show no meaningful position change
      * the operator interrupts (Ctrl+C) - partial evidence is still returned
      * the capture device or the model becomes unusable
    """
    try:
        camera = ctx.hardware.capture()
    except Exception as exc:
        return fail(f"Capture unavailable, so exploration cannot be vision-guided: {exc}")

    try:
        decider = build_vision_decider(ctx, MinecraftFrameDecision)
    except Exception as exc:
        return fail(f"Vision model unavailable: {exc}")

    max_cycles = max(1, int(max_cycles))
    cycles: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    sightings_logged: list[dict[str, Any]] = []
    coords_available = True  # degrades to pixel-delta if this goes False
    last_good_coords: dict[str, float] | None = None
    total_distance = 0.0
    stuck_streak = 0
    dispatched_any = False
    stop_reason = f"Reached the {max_cycles}-cycle budget."
    started = time.time()

    # Real health tracking (see _CRITICAL_HEARTS note above). None until
    # the first successful read, so a HUD read failure (creative mode, or
    # an OCR/mask miss) never falsely looks like a heart drop.
    last_hearts: int | None = None
    min_hearts_observed: int | None = None
    critical_health_stop = False

    print("\n" + "=" * 72, flush=True)
    print("  EXPLORE AND MAP", flush=True)
    print("=" * 72, flush=True)
    print(f"  Max cycles : {max_cycles}", flush=True)
    print("  Stop early : press Ctrl+C - evidence so far is kept", flush=True)
    print("=" * 72, flush=True)

    try:
        for cycle in range(1, max_cycles + 1):
            # --- 1. observe (frame + coordinates) ----------------------------
            before_frame = camera.grab(allow_blank=True)
            if before_frame is None:
                stop_reason = ("Capture returned no frame - the device may have "
                               "been taken by another application.")
                print(f"  [cycle {cycle}] {stop_reason}", flush=True)
                break

            before_path = ctx.artifacts.save_frame(before_frame, f"explore-{cycle:03d}-before")
            image_b64 = encode_frame(before_frame)
            if image_b64 is None:
                stop_reason = "Could not JPEG-encode the frame for the vision model."
                break

            before_coords, before_rejected = _read_coords_checked(
                ctx, before_path, last_good_coords)
            if not before_coords.get("ok"):
                coords_available = False
            elif not before_rejected:
                last_good_coords = {k: before_coords[k] for k in ("x", "y", "z")}

            # Real health check, same pattern as combat_tools.py: read
            # BEFORE deciding this cycle's moves so a detected drop can be
            # injected as an override the model actually sees.
            hud = read_survival_hud_impl(ctx, frame_path=before_path)
            current_hearts = hud.get("full_hearts") if hud.get("ok") else None
            health_line = ""
            if current_hearts is not None:
                if last_hearts is not None and current_hearts < last_hearts:
                    health_line = (
                        f"REAL HEALTH DROP DETECTED: {last_hearts} -> "
                        f"{current_hearts} hearts (measured off the HUD). An "
                        f"unseen threat may be attacking - STOP exploring "
                        f"forward and move AWAY from your last heading this "
                        f"cycle.\n")
                else:
                    health_line = f"Current health: {current_hearts} hearts.\n"
                last_hearts = current_hearts
                min_hearts_observed = (current_hearts if min_hearts_observed is None
                                       else min(min_hearts_observed, current_hearts))

            # --- 2. decide ----------------------------------------------------
            prompt = (
                f"{EXPLORATION_SYSTEM_PROMPT}\n"
                f"# Session state\n"
                f"Cycle: {cycle} of {max_cycles}\n"
                f"Consecutive no-progress cycles: {stuck_streak}\n"
                f"{health_line}\n"
                f"{format_history(history)}\n\n"
                f"Read the attached live frame and return your decision.")

            try:
                decision = decide(decider, prompt, image_b64)
            except Exception as exc:
                print(f"  [cycle {cycle}] vision model error: {exc}", flush=True)
                history.append({"cycle": cycle, "moves": "none", "delta": None,
                                "outcome": f"model error: {exc}"})
                time.sleep(1.0)
                continue

            print(f"\n--- cycle {cycle}/{max_cycles} " + "-" * 40, flush=True)
            print(f"  Scene     : {decision.scene_state}", flush=True)
            if decision.nearby_entities:
                summary = ", ".join(
                    f"{e.kind}({e.category}/{e.direction}/{e.distance_estimate}"
                    f"{'/THREAT' if e.is_threat else ''})"
                    for e in decision.nearby_entities)
                print(f"  Nearby    : {summary}", flush=True)
            print(f"  Thinking  : {decision.reasoning}", flush=True)

            # --- 3. auto-log notable sightings (structures/mobs), coord-keyed --
            logged_this_cycle: list[dict[str, Any]] = []
            if (auto_log_sightings and coords_available
                    and before_coords.get("ok") and not before_rejected):
                for entity in decision.nearby_entities:
                    if entity.category == "structure" or entity.is_threat:
                        rec = record_location_impl(
                            ctx, label=entity.kind,
                            x=before_coords["x"], y=before_coords["y"], z=before_coords["z"],
                            notes=f"category={entity.category} direction={entity.direction} "
                                  f"distance={entity.distance_estimate} cycle={cycle}")
                        if rec.get("ok"):
                            logged_this_cycle.append(rec["entry"])
                            sightings_logged.append(rec["entry"])

            # --- 4. choose the moves --------------------------------------------
            # Hard safety override: a critical health drop is NOT left to the
            # model - the loop itself forces a retreat move, same as
            # combat_tools.py's fix, and STOPS the exploration afterward
            # rather than continuing to wander at near-death health.
            if current_hearts is not None and current_hearts <= _CRITICAL_HEARTS:
                critical_health_stop = True
                moves = [MinecraftMove(
                    action="move", direction="down", duration=1.2,
                    purpose=f"FORCED RETREAT: health at {current_hearts} hearts "
                            f"(<= critical floor {_CRITICAL_HEARTS}), overriding "
                            f"the model's own plan.")]
                print(f"  !! Forcing retreat and stopping - health at or "
                      f"below critical floor ({current_hearts} <= "
                      f"{_CRITICAL_HEARTS})", flush=True)
            else:
                moves = list(decision.moves)
                if decision.scene_state == "cutscene_or_loading" and not moves:
                    moves = [MinecraftMove(action="wait", duration=1.0,
                                           purpose="Wait out a loading/transition screen.")]
                elif decision.scene_state == "menu_or_prompt" and not moves:
                    moves = [MinecraftMove(action="interact", button="a",
                                           purpose="Dismiss the prompt/menu.")]
                elif not moves:
                    moves = [MinecraftMove(
                        action="look", direction="right", duration=0.6,
                        purpose="Model returned no move; look around before continuing.")]

            # --- 5. act -----------------------------------------------------
            # SIMULTANEOUS dispatch: move+look fire together in one GIMX
            # call, matching how a real player turns their head while
            # walking instead of stopping to look then walking.
            for move in moves[:3]:
                print(f"  Act       : {move.action} dir={move.direction} "
                      f"dur={move.duration:.2f}s - {move.purpose}", flush=True)
            outcomes = execute_minecraft_moves(ctx, moves[:3])
            dispatched_moves: list[dict[str, Any]] = []
            for move, outcome in zip(moves[:3], outcomes):
                outcome["purpose"] = move.purpose
                dispatched_moves.append(outcome)
                dispatched_any = dispatched_any or bool(outcome.get("dispatched"))

            # --- 6. re-observe and measure progress -----------------------------
            time.sleep(max(0.0, float(settle_after_move)))
            after_frame = camera.grab(allow_blank=True)
            after_path = (ctx.artifacts.save_frame(after_frame, f"explore-{cycle:03d}-after")
                          if after_frame is not None else None)

            distance_moved = None
            suspect_ocr_jump = False
            after_coords = None
            if coords_available:
                after_coords, after_rejected = _read_coords_checked(
                    ctx, after_path, last_good_coords)
                if after_rejected:
                    suspect_ocr_jump = True
                    # Deliberately NOT updating last_good_coords here - a
                    # rejected read must not become the new anchor, or one
                    # bad OCR read would drag the anchor to the bad value
                    # and make every SUBSEQUENT good read look like the
                    # anomaly instead.
                elif not after_coords.get("ok"):
                    coords_available = False  # HUD read failed - degrade below
                else:
                    # after_coords is good regardless of whether before_coords
                    # was rejected - always advance the anchor on a trusted
                    # read, or a single bad before-read would leave the
                    # anchor stale and risk falsely rejecting the NEXT
                    # cycle's perfectly good read against it.
                    last_good_coords = {k: after_coords[k] for k in ("x", "y", "z")}
                    if before_coords.get("ok") and not before_rejected:
                        distance_moved = _distance(
                            before_coords["x"], before_coords["y"], before_coords["z"],
                            after_coords["x"], after_coords["y"], after_coords["z"])

            pixel_delta = None
            if distance_moved is None and not suspect_ocr_jump:
                # Degraded mode: coordinate HUD unreadable this cycle (or from
                # here on). Pixel delta is a weaker signal (see module
                # docstring) but keeps the loop useful instead of failing.
                pixel_delta = frame_delta(ctx, before_frame, after_frame)

            was_move = any(m.action == "move" for m in moves[:3])
            if was_move:
                if suspect_ocr_jump:
                    # Neither confirms nor denies progress - treat as a
                    # no-op for stuck detection rather than guessing either way.
                    progressed = True
                elif distance_moved is not None:
                    progressed = distance_moved >= _MOVE_DISTANCE_FLOOR
                    total_distance += distance_moved
                elif pixel_delta is not None:
                    progressed = pixel_delta >= 1.0
                else:
                    progressed = False
                stuck_streak = 0 if progressed else stuck_streak + 1
            else:
                progressed = True  # look/wait cycles don't count toward stuck detection

            move_labels = ", ".join(
                str(m.get("macro")) + (f"({m.get('direction')})"
                                       if m.get("direction") else "")
                for m in dispatched_moves) or "none"
            progress_desc = (
                f"moved {distance_moved:.2f} blocks" if distance_moved is not None
                else ("discarded implausible coordinate jump (likely OCR misread)"
                      if suspect_ocr_jump
                      else (f"pixel delta {pixel_delta:.2f}" if pixel_delta is not None
                            else "no measurement")))
            print(f"  Progress  : {progress_desc}", flush=True)

            cycles.append({
                "cycle": cycle,
                "frame_before": before_path,
                "frame_after": after_path,
                "before_coords": {k: before_coords.get(k) for k in ("x", "y", "z")}
                                 if before_coords.get("ok") else None,
                "after_coords": {k: after_coords.get(k) for k in ("x", "y", "z")}
                                if after_coords and after_coords.get("ok") else None,
                "distance_moved": None if distance_moved is None else round(distance_moved, 3),
                "suspect_ocr_jump": suspect_ocr_jump,
                "pixel_delta": None if pixel_delta is None else round(pixel_delta, 4),
                "full_hearts": current_hearts,
                "scene_state": decision.scene_state,
                "nearby_entities": [e.model_dump() for e in decision.nearby_entities],
                "sightings_logged": logged_this_cycle,
                "reasoning": decision.reasoning,
                "moves": dispatched_moves,
                "confidence": decision.confidence,
            })
            history.append({
                "cycle": cycle,
                "moves": move_labels,
                "delta": distance_moved if distance_moved is not None else pixel_delta,
                "outcome": ("progressed" if progressed else
                            "no meaningful position change - that attempt achieved nothing"),
            })

            if critical_health_stop:
                stop_reason = (
                    f"Health dropped to/below the critical floor "
                    f"({current_hearts} <= {_CRITICAL_HEARTS} hearts) - forced "
                    f"a retreat move and STOPPED exploring rather than "
                    f"continuing to wander at near-death health. An unseen "
                    f"threat likely caused this; check nearby_entities/"
                    f"is_threat in recent cycles for a candidate.")
                print(f"\n  !! {stop_reason}", flush=True)
                break

            if stuck_streak >= max(2, int(stuck_limit)):
                stop_reason = (
                    f"{stuck_streak} consecutive move cycles produced no "
                    f"meaningful position change. Either input is not "
                    f"reaching the console or the model is blocked by "
                    f"terrain it cannot navigate around. Stopping instead of "
                    f"sending more input into a void.")
                print(f"\n  !! {stop_reason}", flush=True)
                break

            time.sleep(max(0.0, float(cycle_delay)))

    except KeyboardInterrupt:
        stop_reason = ("Interrupted by the operator. Everything observed up to "
                       "this point is kept as evidence.")
        print(f"\n  [stopped] {stop_reason}", flush=True)

    duration = round(time.time() - started, 2)

    print("\n" + "=" * 72, flush=True)
    print(f"  EXPLORATION ENDED after {len(cycles)} cycles in {duration}s", flush=True)
    print(f"  Reason           : {stop_reason}", flush=True)
    print(f"  Total distance   : {round(total_distance, 2)} blocks "
          f"({'coordinate-tracked' if coords_available else 'DEGRADED - pixel delta used'})",
          flush=True)
    print(f"  Sightings logged : {len(sightings_logged)}", flush=True)
    print("=" * 72 + "\n", flush=True)

    payload = ok(
        cycles_run=len(cycles),
        cycles=cycles,
        stop_reason=stop_reason,
        total_distance_blocks=round(total_distance, 2),
        coordinate_tracking_used=coords_available,
        sightings_logged=sightings_logged,
        min_hearts_observed=min_hearts_observed,
        critical_health_stop=critical_health_stop,
        duration_seconds=duration,
        dispatched=dispatched_any,
        caveat=(
            "dispatched=true only means GIMX accepted the events, and the "
            "model's reasoning is its own hypothesis - neither is proof. "
            "total_distance_blocks is only as trustworthy as the coordinate "
            "HUD OCR (see coordinate_tools.py's own caveats); if "
            "coordinate_tracking_used is false, distance/stuck-detection "
            "silently degraded to the weaker pixel-delta signal for part or "
            "all of this run."),
    )
    ctx.artifacts.save_json("exploration-cycles.json", payload)
    return payload


def _explore_and_map(ctx: ToolContext) -> Any:
    def run(max_cycles: int = 30, cycle_delay: float = 0.2,
            stuck_limit: int = 6, auto_log_sightings: bool = True) -> dict[str, Any]:
        return explore_and_map_impl(
            ctx, max_cycles=max_cycles, cycle_delay=cycle_delay,
            stuck_limit=stuck_limit, auto_log_sightings=auto_log_sightings)

    return make_tool(
        run, "explore_and_map",
        "Explore outward on foot, using COORDINATE deltas (not pixel deltas) "
        "to detect being stuck against terrain - falls back to pixel delta "
        "only if the coordinate HUD becomes unreadable. Auto-logs notable "
        "sightings (structures, threats) via record_location as it goes. "
        "Bounded like every other gameplay loop: stops at max_cycles, a "
        "stuck-streak, an operator interrupt, or a hardware/model failure.")


def provide() -> list[ToolSpec]:
    return [
        ToolSpec(name="explore_and_map",
                 description=("Explore outward on foot with coordinate-based stuck "
                              "detection, auto-logging notable sightings as it goes."),
                 tags=["input", "vision", "game"],
                 factory=_explore_and_map, mutates_hardware=True),
    ]
