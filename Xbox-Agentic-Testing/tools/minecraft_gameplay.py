"""minecraft_gameplay.py - closed-loop, vision-guided Minecraft gameplay.

Standalone by design: this module does NOT import from gameplay_vision.py.
It is a separate observe -> decide -> act -> re-observe loop for a different
game with a different control vocabulary (no Magic Marker, no ink gauge),
so sharing code with the Max loop would only couple two things that change
for unrelated reasons. Small helpers (frame encode/delta, axis/button lookup)
are duplicated locally rather than imported.

GOAL FOR THIS LOOP
-------------------
Find the nearest tree, break its logs (hold RT while facing the trunk),
open the inventory, and craft the logs into planks. There is no fixed
"craft_planks" macro: the model is shown the live frame every cycle and
picks its own button presses to navigate whatever crafting UI is on screen,
the same way a human player would.

THE HONESTY RULE STILL APPLIES
-------------------------------
Nothing here returns `success`. Each cycle records the before/after frame
paths and the measured pixel delta; the tool result carries `dispatched`
plus per-cycle evidence. Whether the console actually reacted is decided
later, by the verifier, from those frames.
"""

from __future__ import annotations

import base64
import time
from typing import Any, Literal

from pydantic import BaseModel, Field

from registry import ToolContext, ToolSpec, fail, make_tool, ok

# Same vision budget as the rest of the framework: logs, tree trunks and
# inventory grid icons stay readable well below native 1080p.
_VISION_WIDTH = 1280
_JPEG_QUALITY = 82


# ===========================================================================
# What the model is allowed to decide
# ===========================================================================
class MinecraftMove(BaseModel):
    """One controller action. Several may be returned per cycle."""

    action: Literal[
        "move",
        "look",
        "mine",
        "interact",
        "press_button",
        "wait",
    ] = Field(description="The controller action to execute.")

    direction: Literal["left", "right", "up", "down", "none"] = Field(
        default="none",
        description="Travel direction for move, or camera-turn direction for look.")

    duration: float = Field(
        default=0.8, ge=0.1, le=6.0,
        description="Seconds to hold the stick/trigger/button. For 'mine' "
                    "this is how long RT stays held on the block - a single "
                    "log usually needs several seconds of continuous holding.")

    button: str = Field(
        default="",
        description="interact/press_button override, e.g. 'a', 'x', 'rb', 'lb'. "
                    "Use 'x' to open/close inventory, 'a' to place/select in a "
                    "crafting grid, 'rb'/'lb' to cycle hotbar slots.")

    purpose: str = Field(
        default="",
        description="One line: what this move is meant to achieve on screen.")


class MinecraftFrameDecision(BaseModel):
    """The model's structured reading of one gameplay frame, plus its plan."""

    scene_state: Literal[
        "in_gameplay",
        "menu_or_prompt",
        "cutscene_or_loading",
        "inventory_open",
        "planks_crafted",
        "stuck_or_blocked",
    ] = Field(description="Classification of the current screen.")

    player_visible: bool = Field(
        description="Is the first-person view / hotbar / crosshair visible?")
    tree_visible: bool = Field(
        description="Is a tree trunk visible anywhere in the frame?")
    log_visible: bool = Field(
        description="Is a wood log block visible directly in front of the "
                    "crosshair (close enough to mine)?")

    reasoning: str = Field(
        description="Step-by-step: the immediate goal, why the chosen moves "
                    "should achieve it, and what the previous cycle's delta "
                    "implies about whether the last attempt worked.")

    moves: list[MinecraftMove] = Field(
        default_factory=list,
        description="1-3 moves to run this cycle, in order. Prefer ONE move "
                    "when the situation is uncertain so the next frame shows "
                    "its isolated effect.")

    confidence: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Confidence that these moves make progress toward "
                    "crafting planks.")


# ===========================================================================
# System prompt
# ===========================================================================
MINECRAFT_GAMEPLAY_SYSTEM_PROMPT = """\
You are playing Minecraft (Xbox Bedrock Edition) on a real Xbox One, in
survival first-person view, inside a freshly created world. You control it
through an emulated controller. The attached image is the LIVE screen right
now - reason only from what you can actually see in it.

GOAL: find the nearest tree, break its logs, open the inventory, and craft
the logs into planks. Then stop - do not build, explore further, or fight
anything once planks exist in the inventory/crafting grid.

CONTROLS AVAILABLE TO YOU (as macros)
  move          : left stick, walk/run in `direction` (forward=up, back=down)
  look          : right stick, turn the camera in `direction` to find/face a tree
  mine          : hold RT while facing a block, breaks it after a few seconds
                  of continuous holding. Must be standing close enough that
                  the block is directly under the crosshair.
  interact      : a single button press, default 'a' - use to place/select
                  an item, confirm a crafting recipe, or take a step in a UI
  press_button  : a single button press with an explicit `button`, e.g. 'x'
                  to open/close the inventory, 'rb'/'lb' to cycle hotbar slots
  wait          : do nothing this cycle, for a screen that is still loading

THERE IS NO DEDICATED "CRAFT PLANKS" BUTTON. Once the inventory is open you
must find the crafting grid yourself from what is on screen: usually placing
a log into a crafting slot with 'interact'/'a' automatically fills the grid
with planks, or a recipe book icon can be selected with 'a' after navigating
to it. Read the screen each cycle and adapt - do not assume a fixed number
of presses works.

SCENE STATE RULES
  in_gameplay      : normal first-person view, hotbar/crosshair visible, no
                      menu overlay
  menu_or_prompt    : any overlay, dialog, or prompt is on screen
  cutscene_or_loading: a loading/black/transition screen with no player HUD
  inventory_open    : the inventory/crafting screen is open
  planks_crafted     : planks are visibly present in the inventory or
                        hotbar - THIS IS THE SUCCESS STATE, report it as soon
                        as you can see plank items on screen
  stuck_or_blocked   : repeated attempts are not changing the screen at all

Return your reading of the frame and 1-3 moves for this cycle.
"""


# ===========================================================================
# Frame helpers (duplicated locally - see module docstring)
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
# Move execution - all button/stick names resolved via the shared pad
# ===========================================================================
def execute_minecraft_move(ctx: ToolContext, move: MinecraftMove) -> dict[str, Any]:
    """Dispatch one decided move. Returns what was sent, never a verdict."""
    pad = ctx.hardware.pad()
    action = move.action
    duration = float(move.duration)

    if action == "move":
        heading = move.direction if move.direction != "none" else "up"
        dispatched = pad.stick("left_stick", direction=heading,
                               duration=duration, strength=1.0)
        return {"macro": "move", "direction": heading,
                "duration": duration, "dispatched": bool(dispatched)}

    if action == "look":
        heading = move.direction if move.direction != "none" else "right"
        dispatched = pad.stick("right_stick", direction=heading,
                               duration=duration, strength=0.6)
        return {"macro": "look", "direction": heading,
                "duration": duration, "dispatched": bool(dispatched)}

    if action == "mine":
        # Same hardware-verified trigger semantics as the Magic Marker: RT is
        # a 0..32767 axis on XOnePad, not 0..255, so pad.hold("rt", ...)
        # (which resolves to the trigger's configured default_press) is what
        # actually registers a full pull on the console.
        dispatched = pad.hold("rt", max(0.3, min(8.0, duration)))
        return {"macro": "mine", "button": "rt",
                "duration": duration, "dispatched": bool(dispatched)}

    if action == "interact":
        button = move.button or "a"
        dispatched = pad.press(button, duration=min(duration, 0.30))
        return {"macro": "interact", "button": button,
                "dispatched": bool(dispatched)}

    if action == "press_button":
        button = move.button or "x"
        dispatched = pad.press(button, duration=min(duration, 0.30))
        return {"macro": "press_button", "button": button,
                "dispatched": bool(dispatched)}

    if action == "wait":
        time.sleep(min(duration, 3.0))
        return {"macro": "wait", "duration": duration, "dispatched": False}

    return {"macro": action, "dispatched": False,
            "error": f"'{action}' is not an implemented Minecraft move."}


# ===========================================================================
# The decision model
# ===========================================================================
def _build_decider(ctx: ToolContext) -> Any:
    """A structured, vision-capable runnable that returns MinecraftFrameDecision."""
    from llm import LLMFactory, structured

    factory = LLMFactory(ctx.settings)
    provider = factory.default_provider
    if not factory.supports_vision(provider):
        raise RuntimeError(
            f"LLM provider '{provider}' is not marked supports_vision in "
            f"settings.yaml, so it cannot look at gameplay frames. Vision-"
            f"guided play needs a multimodal model (anthropic / openai / "
            f"google).")

    return structured(factory.build(provider=provider), MinecraftFrameDecision)


def _decide(decider: Any, prompt: str, image_b64: str) -> MinecraftFrameDecision:
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
def minecraft_gameplay_loop(
    ctx: ToolContext,
    goal: str = ("Find the nearest tree, break its logs by holding RT while "
                 "facing the trunk, then open the inventory and craft the "
                 "logs into planks."),
    max_cycles: int = 45,
    cycle_delay: float = 0.4,
    settle_after_move: float = 0.6,
    stuck_limit: int = 6,
) -> dict[str, Any]:
    """Play Minecraft by looking at every frame and deciding what to do.

    Runs until one of these ends it:
      * `scene_state == "planks_crafted"` is SEEN - the success state
      * `max_cycles` decision cycles have run
      * the operator interrupts (Ctrl+C) - partial evidence is still returned
      * the capture device or the model becomes unusable
      * `stuck_limit` consecutive cycles produce no visual change at all

    Returns per-cycle evidence: the before/after frame path, the measured
    pixel delta, the model's reading of the scene, and the moves dispatched.
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

    stuck_streak = 0
    dispatched_any = False
    planks_crafted: dict[str, Any] | None = None
    stop_reason = f"Reached the {max_cycles}-cycle budget."
    started = time.time()

    print("\n" + "=" * 72, flush=True)
    print("  VISION-GUIDED MINECRAFT GAMEPLAY", flush=True)
    print("=" * 72, flush=True)
    print(f"  Goal       : {goal}", flush=True)
    print(f"  Max cycles : {max_cycles}", flush=True)
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

            before_path = ctx.artifacts.save_frame(before, f"minecraft-{cycle:03d}-before")
            if before_path:
                frames.append(before_path)

            image_b64 = _encode_frame(before)
            if image_b64 is None:
                stop_reason = "Could not JPEG-encode the frame for the vision model."
                break

            # --- 2. decide --------------------------------------------------
            prompt = (
                f"{MINECRAFT_GAMEPLAY_SYSTEM_PROMPT}\n"
                f"# Session state\n"
                f"Goal: {goal}\n"
                f"Cycle: {cycle} of {max_cycles}\n"
                f"Consecutive no-progress cycles: {stuck_streak}\n\n"
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
            print(f"  Tree seen : {decision.tree_visible}", flush=True)
            print(f"  Log seen  : {decision.log_visible}", flush=True)
            print(f"  Thinking  : {decision.reasoning}", flush=True)

            # --- 3. terminal scene state -------------------------------------
            if decision.scene_state == "planks_crafted":
                planks_crafted = {
                    "cycle": cycle,
                    "reasoning": decision.reasoning,
                    "frame_path": before_path,
                }
                print(f"  >> PLANKS_CRAFTED: {decision.reasoning}", flush=True)
                cycles.append({
                    "cycle": cycle,
                    "frame_before": before_path,
                    "frame_after": before_path,
                    "delta": None,
                    "scene_state": decision.scene_state,
                    "reasoning": decision.reasoning,
                    "moves": [],
                    "confidence": decision.confidence,
                })
                stop_reason = f"planks_crafted observed at cycle {cycle}."
                break

            # --- 4. choose the moves ----------------------------------------
            moves = list(decision.moves)
            if decision.scene_state == "cutscene_or_loading" and not moves:
                moves = [MinecraftMove(action="wait", duration=1.0,
                                       purpose="Wait out a loading/transition screen.")]
            elif decision.scene_state == "menu_or_prompt" and not moves:
                moves = [MinecraftMove(action="interact", button="a",
                                       purpose="Dismiss the prompt/menu.")]
            elif not moves:
                # No decision is still a decision: turn to look for a tree
                # rather than burning a cycle standing still.
                moves = [MinecraftMove(
                    action="look", direction="right", duration=0.6,
                    purpose="Model returned no move; look around for a tree.")]

            # --- 5. act -----------------------------------------------------
            dispatched_moves: list[dict[str, Any]] = []
            for move in moves[:3]:
                print(f"  Act       : {move.action} dir={move.direction} "
                      f"dur={move.duration:.2f}s button={move.button!r} "
                      f"- {move.purpose}", flush=True)
                outcome = execute_minecraft_move(ctx, move)
                outcome["purpose"] = move.purpose
                dispatched_moves.append(outcome)
                dispatched_any = dispatched_any or bool(outcome.get("dispatched"))

            # --- 6. re-observe and measure -----------------------------------
            time.sleep(max(0.0, float(settle_after_move)))
            after = camera.grab(allow_blank=True)
            after_path = (ctx.artifacts.save_frame(after, f"minecraft-{cycle:03d}-after")
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
                "tree_visible": decision.tree_visible,
                "log_visible": decision.log_visible,
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
                    f"model cannot find/reach a tree. Stopping instead of "
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
    print(f"  MINECRAFT GAMEPLAY LOOP ENDED after {len(cycles)} cycles in {duration}s",
          flush=True)
    print(f"  Reason      : {stop_reason}", flush=True)
    print("=" * 72 + "\n", flush=True)

    payload = ok(
        goal=goal,
        cycles_run=len(cycles),
        cycles=cycles,
        frames=frames,
        frame_path=frames[-1] if frames else None,
        stop_reason=stop_reason,
        planks_crafted=bool(planks_crafted),
        planks_evidence=planks_crafted,
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
    ctx.artifacts.save_json("minecraft-gameplay-cycles.json", payload)
    return payload


def vision_guided_minecraft_gameplay_impl(
    ctx: ToolContext,
    goal: str = ("Find the nearest tree, break its logs by holding RT while "
                 "facing the trunk, then open the inventory and craft the "
                 "logs into planks."),
    max_cycles: int = 45,
    cycle_delay: float = 0.4,
) -> dict[str, Any]:
    return minecraft_gameplay_loop(ctx, goal=goal, max_cycles=max_cycles,
                                    cycle_delay=cycle_delay)


def _vision_guided_minecraft_gameplay(ctx: ToolContext) -> Any:
    def run(goal: str = ("Find the nearest tree, break its logs by holding "
                         "RT while facing the trunk, then open the "
                         "inventory and craft the logs into planks."),
            max_cycles: int = 45,
            cycle_delay: float = 0.4) -> dict[str, Any]:
        return vision_guided_minecraft_gameplay_impl(
            ctx, goal=goal, max_cycles=max_cycles, cycle_delay=cycle_delay)

    return make_tool(
        run, "vision_guided_minecraft_gameplay",
        "Play Minecraft closed-loop with a VISION model: every cycle it "
        "looks at the live frame and decides moves (move, look, mine, "
        "interact, press_button, wait) to find a tree, break its logs, and "
        "craft planks. Keeps playing until planks are seen, max_cycles is "
        "reached, or nothing on screen changes any more. Returns a "
        "per-cycle log of frames, deltas and reasoning as evidence.")


def provide() -> list[ToolSpec]:
    return [
        ToolSpec(name="vision_guided_minecraft_gameplay",
                 description=("Closed-loop vision gameplay for Minecraft: look at "
                              "the live frame every cycle, find a tree, break its "
                              "logs, open the inventory, and craft planks."),
                 tags=["input", "vision", "game"],
                 factory=_vision_guided_minecraft_gameplay, mutates_hardware=True),
    ]
