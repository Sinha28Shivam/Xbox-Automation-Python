"""
gameplay_vision.py - closed-loop, vision-guided gameplay.

WHY THIS EXISTS
---------------
The first version of `vision_guided_gameplay` decided what to do by grepping
OCR text for words like "pillar" or "press a". That is not vision: an in-game
frame of Max standing at the brink of a chasm contains no text at all, so the
tool always fell through to "move right" and walked him into the pit.

This module does the thing the name promises. Every cycle:

    capture the live frame -> show it to a multimodal model -> get a structured
    decision (what is on screen, what the obstacle is, which controller moves
    to run) -> dispatch those moves -> capture again -> measure the delta

The model sees a real screenshot each cycle, so it decides WHERE to draw, HOW
to draw, when to destroy a drawing, when to jump, and when to keep advancing -
which is exactly the loop a human player runs.

THE HONESTY RULE STILL APPLIES
------------------------------
Nothing here returns `success`. Each cycle records the before/after frame paths
and the measured pixel delta, and the tool result carries `dispatched` plus the
per-cycle evidence. Whether the console actually reacted is decided later, by
the verifier, from those frames. A model saying "I drew a pillar" is a
hypothesis, not proof - so the model's own words are stored as `reasoning`,
never as a verdict.

MAGIC MARKER AIMING
-------------------
In Max: The Curse of Brotherhood the marker cursor starts at Max and is steered
with the left stick while RT (aim) and A (draw) are held. There is no absolute
cursor addressing, so asking a model for screen pixel coordinates would be
false precision. Instead the model supplies a stick VECTOR (aim_x, aim_y in
-1..1) plus a hold duration, which is the control the hardware actually
exposes, and can express any angle - not just the four cardinal directions the
old code allowed.
"""

from __future__ import annotations

import base64
import time
from typing import Any, Literal

from pydantic import BaseModel, Field

from registry import ToolContext, fail, ok

# Vision models cost tokens per pixel, and Max's silhouette, the glowing marker
# nodes and the terrain edges all stay readable well below native 1080p.
# 1280px wide is the same budget the rest of the framework uses.
_VISION_WIDTH = 1280
_JPEG_QUALITY = 82


# ===========================================================================
# What the model is allowed to decide
# ===========================================================================
class GameplayMove(BaseModel):
    """One controller macro. Several may be returned per cycle."""

    action: Literal[
        "move",
        "jump",
        "running_jump",
        "edge_jump_grab",
        "climb_or_pull_up",
        "swing_and_jump",
        "draw_marker",
        "destroy_drawing",
        "push_pull",
        "interact",
        "advance_prompt",
        "wait",
    ] = Field(description="The controller macro to execute.")

    direction: Literal["left", "right", "up", "down", "none"] = Field(
        default="none",
        description="Travel direction for move/jump/push_pull macros.")

    duration: float = Field(
        default=0.8, ge=0.1, le=6.0,
        description="Seconds to hold the stick or the drawing stroke. For "
                    "draw_marker: 1.5-2.0 raises an earth pillar, but a TREE "
                    "BRANCH needs 3.0-5.0 so it grows heavy enough to bend "
                    "and FALL. Too short leaves a stub that never drops.")

    settle_after_draw: float = Field(
        default=1.5, ge=0.0, le=6.0,
        description="draw_marker only: seconds to WAIT after the stroke so "
                    "the drawing finishes animating. A branch keeps bending, "
                    "snaps and falls under its own weight after the stroke "
                    "ends - use 2.5-4.0 when waiting for a branch to fall, "
                    "1.0-1.5 for an earth pillar.")

    run_before_jump: float = Field(
        default=0.4, ge=0.05, le=2.0,
        description="running_jump / edge_jump_grab: run-up time before takeoff.")

    air_time: float = Field(
        default=0.7, ge=0.1, le=2.0,
        description="running_jump / edge_jump_grab: airborne time with the "
                    "stick held toward the target ledge.")

    aim_x: float = Field(
        default=0.0, ge=-1.0, le=1.0,
        description="draw_marker only. Left-stick X while drawing. "
                    "-1 = full left, +1 = full right.")

    aim_y: float = Field(
        default=-1.0, ge=-1.0, le=1.0,
        description="draw_marker only. Left-stick Y while drawing. "
                    "-1 = full UP (raise an earth pillar / grow a branch "
                    "upward), +1 = full down.")

    # ---- Cursor aiming -----------------------------------------------------
    # The marker cursor does NOT start on the node: it opens near the middle
    # of the screen and must be STEERED onto the glowing node before the ink
    # is anchored. Measured travel is ~450 px/s at full deflection.
    node_x: float = Field(
        default=0.0, ge=-1.0, le=1.0,
        description="draw_marker/destroy_drawing. Direction to STEER THE "
                    "CURSOR from the middle of the screen onto the glowing "
                    "node (or onto the drawing you want to erase), BEFORE "
                    "drawing. -1 = left, +1 = right. If the node is below "
                    "and right of screen centre use node_x=+0.7, node_y=+0.7.")

    node_y: float = Field(
        default=0.0, ge=-1.0, le=1.0,
        description="draw_marker/destroy_drawing. Vertical cursor steering. "
                    "-1 = up, +1 = DOWN. Nodes sitting on the ground are "
                    "usually BELOW the cursor's start point, so node_y is "
                    "commonly positive.")

    aim_time: float = Field(
        default=0.4, ge=0.0, le=2.0,
        description="Seconds to steer the cursor before anchoring the ink. "
                    "The cursor moves ~450 px/s, so 0.4s ~= 180px of travel. "
                    "Raise it for a node far from the screen centre; 0 skips "
                    "aiming and draws where the cursor already is.")

    button: str = Field(
        default="",
        description="interact/advance_prompt override, e.g. 'a', 'b', 'x', 'y'.")

    purpose: str = Field(
        default="",
        description="One line: what this move is meant to achieve on screen.")


class FrameDecision(BaseModel):
    """The model's structured reading of one gameplay frame, plus its plan."""

    scene_state: Literal[
        "in_gameplay",
        "cutscene_or_loading",
        "death_or_respawn",
        "menu_or_prompt",
        "checkpoint_reached",
        "level_complete",
    ] = Field(description="Classification of the current screen.")

    player_visible: bool = Field(
        description="Is Max (blue hoodie, cap, orange hair) identifiable?")
    player_position: str = Field(
        description="Where Max is, e.g. 'on the ledge left of a wide chasm', "
                    "'hanging from a vine', 'not visible'.")
    terrain_ahead: str = Field(
        description="What lies in the direction of progress: gap width, ledge "
                    "height, platforms, slopes.")
    hazards: str = Field(
        description="Spikes, thorns, pits, falling rocks, enemies - and where.")
    marker_nodes: str = Field(
        description="Glowing Magic Marker nodes visible (orange earth mounds, "
                    "green tree nodes, water nodes), where they are relative "
                    "to Max, and what drawing on each would create. Say "
                    "'none visible' if there are none.")

    obstacle_type: Literal[
        "none", "gap", "high_ledge", "hazard",
        "enemy", "blocked_by_drawing", "unknown",
    ] = Field(default="none",
              description="The single thing currently stopping progress.")

    progress_direction: Literal["left", "right", "up", "down", "none"] = Field(
        default="right",
        description="Which way the level continues from here.")

    reasoning: str = Field(
        description="Step-by-step: the immediate goal, why the chosen moves "
                    "should achieve it, and what the previous cycle's delta "
                    "implies about whether the last attempt worked.")

    checkpoint_evidence: str = Field(
        default="",
        description="If scene_state is checkpoint_reached or level_complete, "
                    "quote the on-screen evidence (banner text, chapter card, "
                    "'Checkpoint' toast, level-complete summary).")

    moves: list[GameplayMove] = Field(
        default_factory=list,
        description="1-3 moves to run this cycle, in order. Prefer ONE move "
                    "when the situation is uncertain so the next frame shows "
                    "its isolated effect.")

    confidence: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Confidence that these moves advance the level.")


# ===========================================================================
# System prompt
# ===========================================================================
GAMEPLAY_SYSTEM_PROMPT = """\
You are playing "Max: The Curse of Brotherhood" on a real Xbox One. You control
it through an emulated controller. The attached image is the LIVE screen right
now - reason only from what you can actually see in it.

CONTROLS AVAILABLE TO YOU (as macros)
  move                : left stick, walk/run in `direction`
  jump                : A
  running_jump        : run `run_before_jump`s, jump, hold direction `air_time`s
  edge_jump_grab      : run to the brink, high jump, reach forward+UP to catch
                        the far ledge or a rope, then pull up. USE THIS FOR GAPS.
  climb_or_pull_up    : hanging on a ledge/rope -> stick UP + A to get on top
  swing_and_jump      : on a rope/vine -> build momentum, release at the peak
  draw_marker         : the Magic Marker. Two separate vectors:
                        (node_x, node_y) STEERS THE CURSOR onto the glowing
                        node, then (aim_x, aim_y) is the STROKE direction.
                        aim_y=-1 grows the stroke UPWARD.
  destroy_drawing     : erase one of YOUR OWN drawings. Also needs
                        (node_x, node_y) to steer the cursor onto the drawing.
  push_pull           : hold B and move `direction` to shift a block/cart
  interact            : B
  advance_prompt      : A (dismiss a cutscene, dialog, or skip prompt)
  wait                : do nothing for `duration`s

AIMING THE MARKER CURSOR - DO THIS OR THE INK LANDS IN MID-AIR
  When the marker opens, the cursor appears near the MIDDLE OF THE SCREEN -
  NOT on the node and NOT on Max. You must steer it onto the glowing node
  with (node_x, node_y) before the ink is anchored.
    * Compare the node's position to the CENTRE of the image.
      node below-and-right of centre -> node_x=+0.7, node_y=+0.7
      node below-and-left  of centre -> node_x=-0.7, node_y=+0.7
      node straight below centre     -> node_x= 0.0, node_y=+1.0
      REMEMBER: node_y is POSITIVE for DOWN. Ground nodes are usually below
      the centre, so node_y is usually POSITIVE.
    * The cursor travels ~450 px/s, so aim_time 0.4s ~= 180px on a 1920-wide
      screen. Node close to centre -> 0.2s. Node near a screen edge -> 0.8s.
  Getting this wrong is the single most common failure: the stroke appears
  somewhere useless. If the previous cycle drew a pillar in the wrong place,
  change (node_x, node_y) and aim_time - do not just repeat the same values.

STANDING ON YOUR OWN PILLAR (the most useful marker technique)
  A pillar rises from the ground UPWARD. To ride one up to a high ledge, Max
  must already be standing ON the node when it grows:
    1. move so Max is standing directly ON the glowing mound.
    2. draw_marker with aim_y=-1.0 - the pillar lifts Max as it grows.
    3. then jump/move onto the ledge you were trying to reach.
  If instead you need a STEPPING STONE, do the opposite: stand BESIDE the
  node, grow the pillar next to Max, then jump onto its top.
  Decide which of the two you need from the terrain, and say which in
  `reasoning`.

HOW TO DECIDE WHERE AND HOW TO DRAW
  The marker only works on GLOWING NODES. Look for them before proposing a draw:
    * orange glowing mound in the dirt/rock -> draws an EARTH PILLAR upward.
      Use aim_y = -1.0 (straight up) to make a step or a wall. Draw the pillar
      IN the gap to make a stepping stone, or UNDER a high ledge to reach it.
    * green glowing tree stump/branch node  -> grows a BRANCH or a swingable
      vine. Aim toward where you need it to reach: aim_x = +1 grows it right;
      aim_x = +0.7 with aim_y = -0.7 grows it up-and-right at 45 degrees.
    * water node -> sprays a jet of water.
  Aim the stroke at the SPACE YOU NEED FILLED, not at Max. A longer `duration`
  makes a longer stroke: 0.6-1.0s is a short step, 1.5-2.5s is a tall pillar.

  A BRANCH TAKES TIME - KEEP DRAWING UNTIL IT FALLS:
  A tree branch is not done when it first appears. Keep the stroke going until
  it is long and heavy enough to bend over and FALL - the falling branch is
  what bridges the gap, forms a ramp, or knocks an obstacle down.
    * earth pillar : duration 1.5-2.0, settle_after_draw 1.0-1.5
    * TREE BRANCH  : duration 3.0-5.0, settle_after_draw 2.5-4.0
  If a branch draw changed nothing, the stroke was TOO SHORT - raise
  `duration` and draw the same node again rather than giving up on it.
  THE 'X' BADGE ON A DRAWING IS NOT AN ORDER TO DESTROY IT:
  Every drawing carries a small blue 'X' badge meaning "erasable". A pillar
  you just drew is almost always the SOLUTION - it exists to be CLIMBED, not
  erased. Only destroy one that genuinely seals the route, and only after you
  have tried climbing it. Two destroys with no effect = stop destroying.

  GETTING ON TOP OF A PILLAR:
  Walking into a pillar's side does nothing. Note WHICH SIDE of Max it is on -
  if it is behind him, move BACK toward it rather than onward - then use
  edge_jump_grab or running_jump (run_before_jump 0.5-0.8). A standing jump
  is usually too weak for a pillar's full height.

  A LEDGE TOO HIGH FOR ANY JUMP - RIDE THE PILLAR UP:
  Stand ON the glowing node and draw upward so the pillar carries Max with it,
  then step off at the top. This also works while already standing on a pillar
  you drew, letting you stack your way up to an otherwise unreachable ledge.

  If your own drawing is now blocking the route, or grew the wrong way, use
  destroy_drawing (X) and draw again with a different aim.
  If NO node is visible, do NOT propose draw_marker - solve it by platforming,
  or move on to reveal more of the level.

WHEN TO JUMP VERSUS WHEN TO DRAW
  small gap / low step        -> jump or running_jump
  wide chasm at your feet     -> edge_jump_grab, or draw a pillar in the gap
  ledge above head height     -> draw an earth pillar under it, or grow a branch
  spikes/thorns in the path   -> never walk into them; go over via pillar/branch
  hanging on something        -> climb_or_pull_up
  rope/vine over a chasm      -> swing_and_jump

ADVANCING THE LEVEL
  Progress in `progress_direction` (usually right). Do not stand still: if two
  cycles in a row produced almost no screen change, the last approach is not
  working - change it. Never repeat a move that just killed you or did nothing.

SCENE STATES - classify honestly, this drives the loop
  in_gameplay        : you control Max and can act
  cutscene_or_loading: no control; return advance_prompt or wait
  menu_or_prompt     : a dialog/prompt is up; return advance_prompt
  death_or_respawn   : Max died / a respawn screen is showing
  checkpoint_reached : a checkpoint banner/toast or a new chapter card is
                       VISIBLE on screen right now. Only say this when you can
                       actually see it, and quote it in checkpoint_evidence.
                       It stops the session, so do not guess.
  level_complete     : the level-complete/summary screen is visible

Return 1-3 moves. Prefer a single move when unsure, so the next frame isolates
its effect. Stay alive first, then advance.
"""


# ===========================================================================
# Frame helpers
# ===========================================================================
def _encode_frame(frame: Any) -> str | None:
    """Downscale a BGR numpy frame and return base64 JPEG, or None."""
    try:
        import cv2
    except ImportError:
        return None

    height, width = frame.shape[:2]
    if width > _VISION_WIDTH:
        scale = _VISION_WIDTH / float(width)
        frame = cv2.resize(frame, (_VISION_WIDTH, int(height * scale)),
                           interpolation=cv2.INTER_AREA)
    encoded, buffer = cv2.imencode(
        ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), _JPEG_QUALITY])
    if not encoded:
        return None
    return base64.b64encode(buffer.tobytes()).decode("ascii")


def _frame_delta(ctx: ToolContext, before: Any, after: Any) -> float | None:
    """Mean absolute difference between two frames, via the capture layer."""
    if before is None or after is None:
        return None
    try:
        difference = ctx.hardware.capture_functions()["difference"]
        return float(difference(before, after))
    except Exception:
        try:
            import cv2
            return float(cv2.absdiff(before, after).mean())
        except Exception:
            return None


# ===========================================================================
# Controller plumbing - every name comes from controls.yaml, never a literal
# ===========================================================================
def _axis(pad: Any, stick: str, axis: str) -> str:
    spec = pad.cfg.sticks.get(stick, {}) or {}
    fallback = "lstick x" if axis == "x" else "lstick y"
    return str(spec.get(f"{axis}_axis", fallback))


def _axis_range(pad: Any, stick: str) -> tuple[int, int]:
    spec = pad.cfg.sticks.get(stick, {}) or {}
    return int(spec.get("min", -32768)), int(spec.get("max", 32767))


def _scale_axis(pad: Any, stick: str, value: float) -> int:
    """Map a -1..1 request onto the stick's configured axis range."""
    low, high = _axis_range(pad, stick)
    value = max(-1.0, min(1.0, float(value)))
    return int(round(value * (high if value >= 0 else abs(low))))


def _button_control(pad: Any, name: str) -> str:
    spec = pad.cfg.buttons.get(name, {}) or {}
    return str(spec.get("gimx", name))


def _trigger_control(pad: Any, name: str) -> tuple[str, int]:
    spec = pad.cfg.triggers.get(name, {}) or {}
    # Fallback is 32767, NOT 255: on XOnePad the trigger axis spans 0..32767
    # like the sticks. r2(255) is ~0.8% of a pull - GIMX reports success but
    # the console sees an unpressed trigger, so the Magic Marker never opens.
    return str(spec.get("gimx", "r2")), int(spec.get("default_press", 32767))


# ===========================================================================
# Magic Marker
# ===========================================================================
def _hold(pad: Any, events: list[tuple[str, int]], label: str) -> bool:
    """Assert several controller states in ONE gimx call.

    Multi-call sequences can drop a hold between calls, which silently ruins
    the marker: RT must stay down for the whole gesture or the marker closes
    and the stroke is thrown away. `_send_events` exists for exactly this.
    """
    sender = getattr(pad, "_send_events", None)
    if callable(sender):
        return bool(sender(events, label))
    okay = True
    for control, value in events:
        okay = bool(pad._send_event(control, value, label)) and okay
    return okay


def draw_marker_stroke(ctx: ToolContext, aim_x: float = 0.0, aim_y: float = -1.0,
                       duration: float = 1.2, node_x: float = 0.0,
                       node_y: float = 0.0, aim_time: float = 0.4,
                       settle_after_draw: float = 1.5) -> dict[str, Any]:
    """Draw with the Magic Marker: OPEN -> AIM -> ANCHOR -> STROKE -> COMMIT.

    Hardware-verified on this rig (see MARKER_FINDINGS.md). The AIM phase is
    the part that used to be missing: the cursor opens near the middle of the
    screen, NOT on the node, so pressing A immediately anchored the ink at the
    cursor's rest position. Measured rest was (967,534) while the target glow
    sat at (1246,891) - about 450px away. It only ever appeared to work
    because the game auto-snaps ink to a NEARBY node; outside snap range it
    drew in mid-air.

    Aiming always uses FULL deflection: measured gain is ~450 px/s at
    strength 1.0 but only ~17 px/s at 0.4, so a weak push barely moves the
    cursor at all.
    """
    pad = ctx.hardware.pad()
    rt_control, rt_value = _trigger_control(pad, "rt")
    a_control = _button_control(pad, "a")
    x_axis = _axis(pad, "left_stick", "x")
    y_axis = _axis(pad, "left_stick", "y")

    # A zero vector would hold RT+A and draw nothing at all, which reads on the
    # report as "the marker is broken". Default to straight up instead.
    if abs(aim_x) < 0.05 and abs(aim_y) < 0.05:
        aim_x, aim_y = 0.0, -1.0

    x_value = _scale_axis(pad, "left_stick", aim_x)
    y_value = _scale_axis(pad, "left_stick", aim_y)
    # Up to 6s: a tree branch must grow far enough to become heavy and then
    # physically FALL, which takes far longer than raising a short pillar.
    hold = max(0.3, min(6.0, float(duration)))
    aim_hold = max(0.0, min(2.0, float(aim_time)))
    settle = max(0.35, min(6.0, float(settle_after_draw)))
    centre = [(x_axis, 0), (y_axis, 0)]

    # 1. Open the marker. RT spans 0..32767; anything under ~1023 does nothing.
    dispatched = _hold(pad, [(rt_control, rt_value)] + centre, "marker:open")
    time.sleep(0.45)

    # 2. Steer the cursor onto the node, at FULL deflection.
    aimed = False
    if aim_hold > 0.05 and (abs(node_x) >= 0.05 or abs(node_y) >= 0.05):
        norm = max(abs(float(node_x)), abs(float(node_y))) or 1.0
        nx = _scale_axis(pad, "left_stick", float(node_x) / norm)
        ny = _scale_axis(pad, "left_stick", float(node_y) / norm)
        print(f"  -> [MARKER] aim cursor ({node_x:+.2f},{node_y:+.2f}) "
              f"for {aim_hold:.2f}s (~{aim_hold * 450:.0f}px)", flush=True)
        _hold(pad, [(rt_control, rt_value), (x_axis, nx), (y_axis, ny)],
              "marker:aim")
        time.sleep(aim_hold)
        _hold(pad, [(rt_control, rt_value)] + centre, "marker:aim_settle")
        time.sleep(0.20)
        aimed = True

    # 3. Anchor the ink where the cursor now is, then 4. draw the stroke.
    print(f"  -> [MARKER] anchor A, stroke ({aim_x:+.2f},{aim_y:+.2f}) "
          f"for {hold:.2f}s", flush=True)
    _hold(pad, [(rt_control, rt_value), (a_control, 1)] + centre,
          "marker:anchor")
    time.sleep(0.35)
    _hold(pad, [(rt_control, rt_value), (a_control, 1),
                (x_axis, x_value), (y_axis, y_value)], "marker:stroke")
    time.sleep(hold)

    # 5. Commit, then close. A is 1/0 - never 255.
    _hold(pad, [(rt_control, rt_value), (a_control, 1)] + centre,
          "marker:stroke_end")
    time.sleep(0.15)
    _hold(pad, [(rt_control, rt_value), (a_control, 0)] + centre,
          "marker:commit")
    time.sleep(0.25)
    _hold(pad, [(rt_control, 0), (a_control, 0)] + centre, "marker:close")
    # Let the drawing SETTLE before the next frame is judged: a grown branch
    # keeps bending, snaps and falls under its own weight after the stroke
    # ends, and a pillar finishes rising. Measuring too early captures the
    # mid-animation state.
    time.sleep(settle)

    return {
        "macro": "draw_marker",
        "settle_after_draw": settle,
        "aim_x": round(float(aim_x), 3),
        "aim_y": round(float(aim_y), 3),
        "node_x": round(float(node_x), 3),
        "node_y": round(float(node_y), 3),
        "aim_time": aim_hold,
        "cursor_aimed": aimed,
        "duration": hold,
        "dispatched": bool(dispatched),
    }


def destroy_marker_drawing(ctx: ToolContext, node_x: float = 0.0,
                           node_y: float = 0.0, aim_time: float = 0.4,
                           presses: int = 1) -> dict[str, Any]:
    """Erase a drawing: hold RT -> aim at it -> tap X -> release RT.

    A bare `pad.press("x")` cannot work, and that is what this replaces. Two
    hardware-verified requirements it was missing:

      1. RT must stay HELD. X only erases while the marker is OPEN; pressing
         X during normal gameplay is an unrelated context action.
      2. The cursor must physically sit on the drawing. Unlike drawing there
         is NO auto-snap when erasing - which is exactly why destroy failed
         for a long time while draw succeeded with identical aiming.
    """
    pad = ctx.hardware.pad()
    rt_control, rt_value = _trigger_control(pad, "rt")
    x_button = _button_control(pad, "x")
    x_axis = _axis(pad, "left_stick", "x")
    y_axis = _axis(pad, "left_stick", "y")
    aim_hold = max(0.0, min(2.0, float(aim_time)))
    centre = [(x_axis, 0), (y_axis, 0)]

    dispatched = _hold(pad, [(rt_control, rt_value)] + centre, "destroy:open")
    time.sleep(0.45)

    aimed = False
    if aim_hold > 0.05 and (abs(node_x) >= 0.05 or abs(node_y) >= 0.05):
        norm = max(abs(float(node_x)), abs(float(node_y))) or 1.0
        nx = _scale_axis(pad, "left_stick", float(node_x) / norm)
        ny = _scale_axis(pad, "left_stick", float(node_y) / norm)
        print(f"  -> [DESTROY] aim cursor ({node_x:+.2f},{node_y:+.2f}) "
              f"for {aim_hold:.2f}s", flush=True)
        _hold(pad, [(rt_control, rt_value), (x_axis, nx), (y_axis, ny)],
              "destroy:aim")
        time.sleep(aim_hold)
        aimed = True

    # Hold the aim through the press: re-centring first can let the cursor
    # drift off the drawing before X registers.
    taps = max(1, min(5, int(presses)))
    print(f"  -> [DESTROY] press X x{taps} (RT still held)", flush=True)
    for _ in range(taps):
        _hold(pad, [(rt_control, rt_value), (x_button, 1)], "destroy:x_down")
        time.sleep(0.25)
        _hold(pad, [(rt_control, rt_value), (x_button, 0)], "destroy:x_up")
        time.sleep(0.30)

    _hold(pad, [(rt_control, 0), (x_button, 0)] + centre, "destroy:close")
    time.sleep(0.35)

    return {
        "macro": "destroy_drawing",
        "button": x_button,
        "node_x": round(float(node_x), 3),
        "node_y": round(float(node_y), 3),
        "aim_time": aim_hold,
        "cursor_aimed": aimed,
        "presses": taps,
        "dispatched": bool(dispatched),
    }


# ===========================================================================
# Move execution
# ===========================================================================
def execute_move(ctx: ToolContext, move: GameplayMove) -> dict[str, Any]:
    """Dispatch one decided macro. Returns what was sent, never a verdict."""
    pad = ctx.hardware.pad()
    action = move.action
    duration = float(move.duration)
    x_axis = _axis(pad, "left_stick", "x")
    y_axis = _axis(pad, "left_stick", "y")

    def horizontal(default: str = "right") -> int:
        heading = move.direction if move.direction in ("left", "right") else default
        return _scale_axis(pad, "left_stick", 1.0 if heading == "right" else -1.0)

    if action == "move":
        heading = move.direction if move.direction != "none" else "right"
        moved = pad.stick("left_stick", direction=heading,
                          duration=duration, strength=1.0)
        return {"macro": "move", "direction": heading,
                "duration": duration, "dispatched": bool(moved)}

    if action == "jump":
        pressed = pad.press("a", duration=min(duration, 0.25))
        return {"macro": "jump", "button": "a", "dispatched": bool(pressed)}

    if action == "running_jump":
        pad._send_event(x_axis, horizontal(), "run:accelerate")
        time.sleep(float(move.run_before_jump))
        pad.press("a", duration=0.20)
        time.sleep(float(move.air_time))
        pad._send_event(x_axis, 0, "run:release")
        time.sleep(0.20)
        return {"macro": "running_jump", "direction": move.direction,
                "run_before_jump": move.run_before_jump,
                "air_time": move.air_time, "dispatched": True}

    if action == "edge_jump_grab":
        # Sprint to the brink, leap, then reach forward AND up so Max catches
        # the far edge or a rope instead of arcing past it, then hoist up.
        pad._send_event(x_axis, horizontal(), "edge:sprint")
        time.sleep(float(move.run_before_jump))
        pad.press("a", duration=0.22)
        pad._send_event(y_axis, _scale_axis(pad, "left_stick", -0.75),
                        "edge:reach_up")
        time.sleep(float(move.air_time))
        pad._send_event(y_axis, _scale_axis(pad, "left_stick", -1.0),
                        "edge:pull_up")
        pad.press("a", duration=0.15)
        time.sleep(0.35)
        pad._send_event(x_axis, 0, "edge:release_x")
        pad._send_event(y_axis, 0, "edge:release_y")
        time.sleep(0.15)
        return {"macro": "edge_jump_grab", "direction": move.direction,
                "dispatched": True}

    if action == "climb_or_pull_up":
        pad._send_event(y_axis, _scale_axis(pad, "left_stick", -1.0), "climb:up")
        pad.press("a", duration=0.22)
        pad._send_event(x_axis, horizontal(), "climb:forward")
        time.sleep(min(duration, 0.8))
        pad._send_event(y_axis, 0, "climb:release_y")
        pad._send_event(x_axis, 0, "climb:release_x")
        time.sleep(0.15)
        return {"macro": "climb_or_pull_up", "dispatched": True}

    if action == "swing_and_jump":
        pad._send_event(x_axis, _scale_axis(pad, "left_stick", -1.0), "swing:back")
        time.sleep(0.40)
        pad._send_event(x_axis, _scale_axis(pad, "left_stick", 1.0), "swing:forward")
        time.sleep(0.50)
        pad.press("a", duration=0.22)
        time.sleep(0.60)
        pad._send_event(x_axis, 0, "swing:release")
        time.sleep(0.20)
        return {"macro": "swing_and_jump", "dispatched": True}

    if action == "draw_marker":
        return draw_marker_stroke(ctx, aim_x=move.aim_x, aim_y=move.aim_y,
                                  duration=duration,
                                  node_x=move.node_x, node_y=move.node_y,
                                  aim_time=move.aim_time,
                                  settle_after_draw=move.settle_after_draw)

    if action == "destroy_drawing":
        # NOT a bare X press: X only erases while RT holds the marker open,
        # and the cursor has to be steered onto the drawing first.
        return destroy_marker_drawing(ctx, node_x=move.node_x,
                                      node_y=move.node_y,
                                      aim_time=move.aim_time)

    if action == "push_pull":
        b_control = _button_control(pad, "b")
        pad._send_event(b_control, 1, "grip:hold")
        pad._send_event(x_axis, horizontal(), "push:move")
        time.sleep(duration)
        pad._send_event(x_axis, 0, "push:release")
        pad._send_event(b_control, 0, "grip:release")
        time.sleep(0.20)
        return {"macro": "push_pull", "direction": move.direction,
                "dispatched": True}

    if action == "interact":
        button = move.button or "b"
        pressed = pad.press(button, duration=min(duration, 0.30))
        return {"macro": "interact", "button": button,
                "dispatched": bool(pressed)}

    if action == "advance_prompt":
        button = move.button or "a"
        pressed = pad.press(button, duration=0.20)
        time.sleep(1.20)
        return {"macro": "advance_prompt", "button": button,
                "dispatched": bool(pressed)}

    if action == "wait":
        time.sleep(min(duration, 3.0))
        return {"macro": "wait", "duration": duration, "dispatched": False}

    return {"macro": action, "dispatched": False,
            "error": f"'{action}' is not an implemented gameplay macro."}


# ===========================================================================
# The decision model
# ===========================================================================
def _build_decider(ctx: ToolContext) -> Any:
    """A structured, vision-capable runnable that returns FrameDecision."""
    from llm import LLMFactory, structured

    factory = LLMFactory(ctx.settings)
    provider = factory.default_provider
    if not factory.supports_vision(provider):
        raise RuntimeError(
            f"LLM provider '{provider}' is not marked supports_vision in "
            f"settings.yaml, so it cannot look at gameplay frames. Vision-"
            f"guided play needs a multimodal model (anthropic / openai / "
            f"google).")

    return structured(factory.build(provider=provider), FrameDecision)


def _decide(decider: Any, prompt: str, image_b64: str) -> FrameDecision:
    from langchain_core.messages import HumanMessage

    message = HumanMessage(content=[
        {"type": "text", "text": prompt},
        {"type": "image_url",
         "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
    ])
    return decider.invoke([message])


def _format_history(history: list[dict[str, Any]], keep: int = 4) -> str:
    """Recent cycles, so the model can tell whether its last idea worked."""
    if not history:
        return "This is the first cycle - no history yet."
    lines = ["Recent cycles (delta = mean pixel change; near 0 means the "
             "screen did NOT move, so that attempt achieved nothing):"]
    for entry in history[-keep:]:
        delta = entry.get("delta")
        delta_text = "unknown" if delta is None else f"{delta:.2f}"
        lines.append(f"  - cycle {entry['cycle']}: {entry['moves']} "
                     f"-> delta {delta_text} ({entry['outcome']})")
    return "\n".join(lines)


# ===========================================================================
# The loop
# ===========================================================================
def vision_gameplay_loop(
    ctx: ToolContext,
    goal: str = "Advance through the level, drawing with the Magic Marker "
                "where needed, until a checkpoint is reached.",
    max_cycles: int = 40,
    cycle_delay: float = 0.4,
    stop_on_checkpoint: bool = True,
    settle_after_move: float = 0.6,
    stuck_limit: int = 6,
) -> dict[str, Any]:
    """Play the game by looking at every frame and deciding what to do.

    Runs until one of these ends it:
      * a checkpoint / level-complete screen is SEEN (when stop_on_checkpoint)
      * `max_cycles` decision cycles have run
      * the operator interrupts (Ctrl+C) - partial evidence is still returned
      * the capture device or the model becomes unusable
      * `stuck_limit` consecutive cycles produce no visual change at all

    Returns per-cycle evidence: the before/after frame path, the measured pixel
    delta, the model's reading of the scene, and the macros dispatched.
    """
    try:
        camera = ctx.hardware.capture()
    except Exception as exc:
        return fail(f"Capture unavailable, so gameplay cannot be vision-guided: {exc}")

    try:
        decider = _build_decider(ctx)
    except Exception as exc:
        return fail(f"Vision model unavailable: {exc}")

    max_cycles = max(1, int(max_cycles))
    cycles: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    frames: list[str] = []
    deltas: list[float] = []

    deaths = 0
    draws = 0
    destroys = 0
    stuck_streak = 0
    dispatched_any = False
    checkpoint: dict[str, Any] | None = None
    stop_reason = f"Reached the {max_cycles}-cycle budget."
    started = time.time()

    print("\n" + "=" * 72, flush=True)
    print("  VISION-GUIDED GAMEPLAY", flush=True)
    print("=" * 72, flush=True)
    print(f"  Goal       : {goal}", flush=True)
    print(f"  Max cycles : {max_cycles}", flush=True)
    print(f"  Stop at    : "
          f"{'first visible checkpoint' if stop_on_checkpoint else 'cycle budget only'}",
          flush=True)
    print("  Stop early : press Ctrl+C - evidence so far is kept", flush=True)
    print("=" * 72, flush=True)

    try:
        for cycle in range(1, max_cycles + 1):
            # --- 1. observe -------------------------------------------------
            before = camera.grab(allow_blank=True)
            if before is None:
                stop_reason = ("Capture returned no frame - the device may have "
                               "been taken by another application.")
                print(f"  [cycle {cycle}] {stop_reason}", flush=True)
                break

            before_path = ctx.artifacts.save_frame(before, f"play-{cycle:03d}-before")
            if before_path:
                frames.append(before_path)

            image_b64 = _encode_frame(before)
            if image_b64 is None:
                stop_reason = "Could not JPEG-encode the frame for the vision model."
                break

            # --- 2. decide --------------------------------------------------
            prompt = (
                f"{GAMEPLAY_SYSTEM_PROMPT}\n"
                f"# Session state\n"
                f"Goal: {goal}\n"
                f"Cycle: {cycle} of {max_cycles}\n"
                f"Deaths so far: {deaths}\n"
                f"Consecutive no-progress cycles: {stuck_streak}\n"
                f"Marker strokes drawn so far: {draws} (erased {destroys})\n\n"
                f"{_format_history(history)}\n\n"
                f"Read the attached live frame and return your decision."
            )

            try:
                decision = _decide(decider, prompt, image_b64)
            except Exception as exc:
                print(f"  [cycle {cycle}] vision model error: {exc}", flush=True)
                history.append({"cycle": cycle, "moves": "none", "delta": None,
                                "outcome": f"model error: {exc}"})
                time.sleep(1.0)
                continue

            print(f"\n--- cycle {cycle}/{max_cycles} " + "-" * 40, flush=True)
            print(f"  Scene     : {decision.scene_state}", flush=True)
            print(f"  Max       : {decision.player_position}", flush=True)
            print(f"  Ahead     : {decision.terrain_ahead}", flush=True)
            print(f"  Hazards   : {decision.hazards}", flush=True)
            print(f"  Nodes     : {decision.marker_nodes}", flush=True)
            print(f"  Obstacle  : {decision.obstacle_type}", flush=True)
            print(f"  Thinking  : {decision.reasoning}", flush=True)

            # --- 3. terminal scene states -----------------------------------
            if decision.scene_state in ("checkpoint_reached", "level_complete"):
                checkpoint = {
                    "cycle": cycle,
                    "scene_state": decision.scene_state,
                    "evidence": decision.checkpoint_evidence,
                    "frame_path": before_path,
                }
                print(f"  >> {decision.scene_state.upper()}: "
                      f"{decision.checkpoint_evidence}", flush=True)
                cycles.append({
                    "cycle": cycle,
                    "frame_before": before_path,
                    "frame_after": before_path,
                    "delta": None,
                    "scene_state": decision.scene_state,
                    "obstacle_type": decision.obstacle_type,
                    "reasoning": decision.reasoning,
                    "marker_nodes": decision.marker_nodes,
                    "moves": [],
                    "confidence": decision.confidence,
                })
                if stop_on_checkpoint:
                    stop_reason = (
                        f"{decision.scene_state} observed at cycle {cycle}: "
                        f"{decision.checkpoint_evidence or 'no text quoted'}")
                    break
                continue

            # --- 4. choose the moves ----------------------------------------
            moves = list(decision.moves)
            if decision.scene_state == "death_or_respawn":
                deaths += 1
                moves = [GameplayMove(action="advance_prompt", button="a",
                                      purpose="Respawn at the last checkpoint.")]
            elif decision.scene_state in ("cutscene_or_loading", "menu_or_prompt"):
                if not moves:
                    moves = [GameplayMove(action="advance_prompt", button="a",
                                          purpose="Dismiss the prompt/cutscene.")]
            elif not moves:
                # No decision is still a decision: nudge forward one beat rather
                # than burning a cycle standing still.
                heading = (decision.progress_direction
                           if decision.progress_direction != "none" else "right")
                moves = [GameplayMove(
                    action="move", direction=heading, duration=0.7,
                    purpose="Model returned no move; advance to reveal more.")]

            # --- 5. act -----------------------------------------------------
            dispatched_moves: list[dict[str, Any]] = []
            for move in moves[:3]:
                aim = ""
                if move.action == "draw_marker":
                    aim = (f"node=({move.node_x:+.2f},{move.node_y:+.2f})@"
                           f"{move.aim_time:.2f}s "
                           f"stroke=({move.aim_x:+.2f},{move.aim_y:+.2f}) ")
                elif move.action == "destroy_drawing":
                    aim = (f"node=({move.node_x:+.2f},{move.node_y:+.2f})@"
                           f"{move.aim_time:.2f}s ")
                print(f"  Act       : {move.action} dir={move.direction} "
                      f"dur={move.duration:.2f}s {aim}- {move.purpose}",
                      flush=True)
                outcome = execute_move(ctx, move)
                outcome["purpose"] = move.purpose
                dispatched_moves.append(outcome)
                dispatched_any = dispatched_any or bool(outcome.get("dispatched"))
                if move.action == "draw_marker":
                    draws += 1
                elif move.action == "destroy_drawing":
                    destroys += 1

            # --- 6. re-observe and measure ----------------------------------
            time.sleep(max(0.0, float(settle_after_move)))
            after = camera.grab(allow_blank=True)
            after_path = (ctx.artifacts.save_frame(after, f"play-{cycle:03d}-after")
                          if after is not None else None)
            if after_path:
                frames.append(after_path)

            delta = _frame_delta(ctx, before, after)
            if delta is not None:
                deltas.append(delta)
                print(f"  Delta     : {delta:.3f}", flush=True)

            progressed = delta is not None and delta >= 1.0
            stuck_streak = 0 if progressed else stuck_streak + 1
            move_labels = ", ".join(
                str(m.get("macro")) + (f"({m.get('direction')})"
                                       if m.get("direction") else "")
                for m in dispatched_moves) or "none"

            cycles.append({
                "cycle": cycle,
                "frame_before": before_path,
                "frame_after": after_path,
                "delta": None if delta is None else round(delta, 4),
                "scene_state": decision.scene_state,
                "obstacle_type": decision.obstacle_type,
                "player_position": decision.player_position,
                "terrain_ahead": decision.terrain_ahead,
                "hazards": decision.hazards,
                "marker_nodes": decision.marker_nodes,
                "reasoning": decision.reasoning,
                "moves": dispatched_moves,
                "confidence": decision.confidence,
            })
            history.append({
                "cycle": cycle,
                "moves": move_labels,
                "delta": delta,
                "outcome": ("screen moved" if progressed else
                            "screen did NOT move - that attempt achieved nothing"),
            })

            if stuck_streak >= max(2, int(stuck_limit)):
                stop_reason = (
                    f"{stuck_streak} consecutive cycles produced no visual "
                    f"change. Either input is not reaching the console (an "
                    f"unauthenticated GIMX session is the usual cause) or the "
                    f"model cannot solve this obstacle. Stopping instead of "
                    f"sending more input into a void.")
                print(f"\n  !! {stop_reason}", flush=True)
                break

            time.sleep(max(0.0, float(cycle_delay)))

    except KeyboardInterrupt:
        stop_reason = ("Interrupted by the operator. Everything observed up to "
                       "this point is kept as evidence.")
        print(f"\n  [stopped] {stop_reason}", flush=True)

    duration = round(time.time() - started, 2)
    mean_delta = round(sum(deltas) / len(deltas), 4) if deltas else None
    max_delta = round(max(deltas), 4) if deltas else None

    print("\n" + "=" * 72, flush=True)
    print(f"  GAMEPLAY LOOP ENDED after {len(cycles)} cycles in {duration}s",
          flush=True)
    print(f"  Reason      : {stop_reason}", flush=True)
    print(f"  Marker draws: {draws}   erased: {destroys}   deaths: {deaths}",
          flush=True)
    print(f"  Mean delta  : {mean_delta}   max delta: {max_delta}", flush=True)
    print("=" * 72 + "\n", flush=True)

    payload = ok(
        goal=goal,
        cycles_run=len(cycles),
        cycles=cycles,
        frames=frames,
        frame_path=frames[-1] if frames else None,
        stop_reason=stop_reason,
        checkpoint_reached=bool(checkpoint),
        checkpoint=checkpoint,
        marker_draws=draws,
        drawings_destroyed=destroys,
        deaths=deaths,
        mean_delta=mean_delta,
        max_delta=max_delta,
        observed_change=bool(max_delta is not None and max_delta >= 1.0),
        duration_seconds=duration,
        dispatched=dispatched_any,
        caveat=(
            "dispatched=true only means GIMX accepted the events, and the "
            "model's reasoning is its own hypothesis - neither is proof. The "
            "per-cycle frame pairs and measured deltas are the evidence; the "
            "verifier decides what they show."),
    )
    ctx.artifacts.save_json("gameplay-cycles.json", payload)
    return payload
