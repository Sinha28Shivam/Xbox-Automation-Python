"""gameplay_engine.py - the game-AGNOSTIC half of closed-loop vision gameplay.

Extracted from gameplay_vision.py (Max) and minecraft_gameplay.py (Minecraft)
after those two files reached ~80% line-for-line duplication: frame capture/
encode, the observe->decide->act->re-observe loop shape, the vision-model
decider, and stuck-cycle detection were being copy-pasted per game, which
meant every NEW game would repeat the same plumbing (and risk silently
drifting from it).

A game plugs into this engine by building a `GameProfile` - one bundle of
small hooks covering exactly the points where Max and Minecraft actually
differed (move vocabulary, system prompt, per-scene fallback moves, stuck-
counter grouping, terminal/success detection, and end-of-run summary keys).
Nothing generic about the loop lives in a profile, and nothing game-specific
lives here.

THE HONESTY RULE STILL APPLIES
-------------------------------
`run_gameplay_loop` never returns `success`. Each cycle records the before/
after frame paths and the measured pixel delta; the caller decides what that
evidence proves.
"""

from __future__ import annotations

import base64
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from pydantic import BaseModel

from registry import ToolContext, fail, ok

# Vision models cost tokens per pixel; 1280px wide is the budget the whole
# framework uses - game silhouettes, terrain edges and UI icons stay readable
# well below native 1080p.
VISION_WIDTH = 1280
JPEG_QUALITY = 82


# ===========================================================================
# Frame helpers (the genuinely game-agnostic ones)
# ===========================================================================
def encode_frame(frame: Any) -> str | None:
    """Downscale a BGR numpy frame and return base64 JPEG, or None."""
    try:
        import cv2
    except ImportError:
        return None

    height, width = frame.shape[:2]
    if width > VISION_WIDTH:
        scale = VISION_WIDTH / float(width)
        frame = cv2.resize(frame, (VISION_WIDTH, int(height * scale)),
                           interpolation=cv2.INTER_AREA)
    encoded, buffer = cv2.imencode(
        ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    if not encoded:
        return None
    return base64.b64encode(buffer.tobytes()).decode("ascii")


def frame_delta(ctx: ToolContext, before: Any, after: Any) -> float | None:
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


def build_vision_decider(ctx: ToolContext, frame_model: type[BaseModel]) -> Any:
    """A structured, vision-capable runnable that returns `frame_model`."""
    from llm import LLMFactory, structured

    factory = LLMFactory(ctx.settings)
    provider = factory.default_provider
    if not factory.supports_vision(provider):
        raise RuntimeError(
            f"LLM provider '{provider}' is not marked supports_vision in "
            f"settings.yaml, so it cannot look at gameplay frames. Vision-"
            f"guided play needs a multimodal model (anthropic / openai / "
            f"google).")

    return structured(factory.build(provider=provider), frame_model)


def decide(decider: Any, prompt: str, image_b64: str) -> Any:
    from langchain_core.messages import HumanMessage

    message = HumanMessage(content=[
        {"type": "text", "text": prompt},
        {"type": "image_url",
         "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
    ])
    return decider.invoke([message])


def format_history(history: list[dict[str, Any]], keep: int = 4) -> str:
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
# The contract each game implements - every field maps to a real divergence
# found while diffing the Max and Minecraft loops, nothing speculative.
# ===========================================================================
@dataclass
class GameProfile:
    key: str                                   # e.g. "max", "minecraft"
    move_model: type[BaseModel]
    frame_model: type[BaseModel]
    system_prompt: str
    default_goal: str
    artifact_prefix: str                        # frame filename prefix
    json_artifact_name: str                      # saved cycles-log filename
    execute_move: Callable[[ToolContext, Any], dict[str, Any]]

    success_states: set[str]
    terminal_flag_key: str                        # payload key, e.g. "planks_crafted"
    terminal_evidence_key: str                     # payload key, e.g. "planks_evidence"
    build_terminal_evidence: Callable[[Any, int, str | None], dict[str, Any]]

    # counters: an open dict the profile owns entirely (deaths, draws, ...)
    initial_counters: Callable[[], dict[str, Any]] = field(default=lambda: {})
    forced_moves: Callable[[Any, dict[str, Any]], list[Any] | None] = field(
        default=lambda decision, counters: None)
    fallback_moves: Callable[[Any], list[Any]] = field(
        default=lambda decision: [])
    on_move_dispatched: Callable[[Any, dict[str, Any]], None] = field(
        default=lambda move, counters: None)

    # stuck-cycle detection: the profile owns the counters and the rule for
    # updating them (Minecraft's mining patience is deliberately asymmetric -
    # see minecraft_gameplay.py - so a generic "reset every other group" rule
    # in this engine would silently change verified behavior).
    update_stuck_counters: Callable[[dict[str, int], list[Any], bool], None] = field(
        default=lambda counters, moves, progressed: counters.__setitem__(
            "stuck_streak", 0 if progressed else counters.get("stuck_streak", 0) + 1))
    stuck_limits: Callable[[int], dict[str, int]] = field(
        default=lambda stuck_limit: {"stuck_streak": max(2, int(stuck_limit))})
    stuck_message: Callable[[str, int], str] = field(
        default=lambda name, value: (
            f"{value} consecutive cycles produced no visual change. Either "
            f"input is not reaching the console or the model cannot solve "
            f"this. Stopping instead of sending more input into a void."))

    extra_session_text: Callable[[dict[str, Any]], str] = field(default=lambda counters: "")
    print_extra: Callable[[Any], None] = field(default=lambda decision: None)
    cycle_extra_fields: Callable[[Any], dict[str, Any]] = field(default=lambda decision: {})
    summarize: Callable[[dict[str, Any]], dict[str, Any]] = field(default=lambda counters: {})

    # Optional SIMULTANEOUS batch dispatch (move+look+attack in one GIMX
    # call, matching how a real player's hands work at once) - preferred
    # over the one-at-a-time execute_move loop below when a profile
    # provides it. None means "use the old sequential path" so existing
    # profiles (Max) are unaffected unless they opt in.
    execute_moves: Callable[[ToolContext, list[Any]], list[dict[str, Any]]] | None = None


# ===========================================================================
# The one shared loop every game runs through
# ===========================================================================
def run_gameplay_loop(
    ctx: ToolContext,
    profile: GameProfile,
    goal: str | None = None,
    max_cycles: int = 40,
    cycle_delay: float = 0.4,
    settle_after_move: float = 0.6,
    stuck_limit: int = 6,
    stop_on_success: bool = True,
) -> dict[str, Any]:
    """Play a game by looking at every frame and deciding what to do.

    Runs until one of these ends it:
      * a scene_state in `profile.success_states` is seen (and stop_on_success)
      * `max_cycles` decision cycles have run
      * the operator interrupts (Ctrl+C) - partial evidence is still returned
      * the capture device or the model becomes unusable
      * one of the profile's stuck counters reaches its limit

    Returns per-cycle evidence: the before/after frame path, the measured
    pixel delta, the model's reading of the scene, and the moves dispatched.
    """
    goal = goal or profile.default_goal

    try:
        camera = ctx.hardware.capture()
    except Exception as exc:
        return fail(f"Capture unavailable, so gameplay cannot be vision-guided: {exc}")

    try:
        decider = build_vision_decider(ctx, profile.frame_model)
    except Exception as exc:
        return fail(f"Vision model unavailable: {exc}")

    max_cycles = max(1, int(max_cycles))
    cycles: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    frames: list[str] = []
    deltas: list[float] = []
    counters: dict[str, Any] = profile.initial_counters()

    dispatched_any = False
    terminal_evidence: dict[str, Any] | None = None
    stop_reason = f"Reached the {max_cycles}-cycle budget."
    started = time.time()

    print("\n" + "=" * 72, flush=True)
    print(f"  VISION-GUIDED GAMEPLAY ({profile.key})", flush=True)
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

            before_path = ctx.artifacts.save_frame(
                before, f"{profile.artifact_prefix}-{cycle:03d}-before")
            if before_path:
                frames.append(before_path)

            image_b64 = encode_frame(before)
            if image_b64 is None:
                stop_reason = "Could not JPEG-encode the frame for the vision model."
                break

            # --- 2. decide --------------------------------------------------
            stuck_text = ", ".join(
                f"{name}={value}" for name, value in counters.items()
                if isinstance(value, int)) or "none"
            prompt = (
                f"{profile.system_prompt}\n"
                f"# Session state\n"
                f"Goal: {goal}\n"
                f"Cycle: {cycle} of {max_cycles}\n"
                f"Consecutive no-progress counters: {stuck_text}\n"
                f"{profile.extra_session_text(counters)}"
                f"\n{format_history(history)}\n\n"
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
            profile.print_extra(decision)
            print(f"  Thinking  : {decision.reasoning}", flush=True)

            # --- 3. terminal (success) scene state ---------------------------
            if decision.scene_state in profile.success_states:
                terminal_evidence = profile.build_terminal_evidence(
                    decision, cycle, before_path)
                print(f"  >> {decision.scene_state.upper()}: "
                      f"{terminal_evidence}", flush=True)
                cycles.append({
                    "cycle": cycle,
                    "frame_before": before_path,
                    "frame_after": before_path,
                    "delta": None,
                    "scene_state": decision.scene_state,
                    "reasoning": decision.reasoning,
                    "moves": [],
                    "confidence": decision.confidence,
                    **profile.cycle_extra_fields(decision),
                })
                if stop_on_success:
                    stop_reason = f"{decision.scene_state} observed at cycle {cycle}."
                    break
                continue

            # --- 4. choose the moves ------------------------------------------
            moves = profile.forced_moves(decision, counters)
            if moves is None:
                moves = list(decision.moves)
                if not moves:
                    moves = profile.fallback_moves(decision)

            # --- 5. act -------------------------------------------------------
            dispatched_moves: list[dict[str, Any]] = []
            for move in moves[:3]:
                print(f"  Act       : {move.action} dir={move.direction} "
                      f"dur={move.duration:.2f}s - {move.purpose}", flush=True)
            if profile.execute_moves is not None:
                # Simultaneous dispatch - see GameProfile.execute_moves note.
                outcomes = profile.execute_moves(ctx, moves[:3])
                for move, outcome in zip(moves[:3], outcomes):
                    outcome["purpose"] = move.purpose
                    dispatched_moves.append(outcome)
                    dispatched_any = dispatched_any or bool(outcome.get("dispatched"))
                    profile.on_move_dispatched(move, counters)
            else:
                for move in moves[:3]:
                    outcome = profile.execute_move(ctx, move)
                    outcome["purpose"] = move.purpose
                    dispatched_moves.append(outcome)
                    dispatched_any = dispatched_any or bool(outcome.get("dispatched"))
                    profile.on_move_dispatched(move, counters)

            # --- 6. re-observe and measure -------------------------------------
            time.sleep(max(0.0, float(settle_after_move)))
            after = camera.grab(allow_blank=True)
            after_path = (ctx.artifacts.save_frame(after, f"{profile.artifact_prefix}-{cycle:03d}-after")
                          if after is not None else None)
            if after_path:
                frames.append(after_path)

            delta = frame_delta(ctx, before, after)
            if delta is not None:
                deltas.append(delta)
                print(f"  Delta     : {delta:.3f}", flush=True)

            progressed = delta is not None and delta >= 1.0
            profile.update_stuck_counters(counters, moves[:3], progressed)

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
                "reasoning": decision.reasoning,
                "moves": dispatched_moves,
                "confidence": decision.confidence,
                **profile.cycle_extra_fields(decision),
            })
            history.append({
                "cycle": cycle,
                "moves": move_labels,
                "delta": delta,
                "outcome": ("screen moved" if progressed else
                            "screen did NOT move - that attempt achieved nothing"),
            })

            for name, limit in profile.stuck_limits(stuck_limit).items():
                value = counters.get(name, 0)
                if value >= limit:
                    stop_reason = profile.stuck_message(name, value)
                    print(f"\n  !! {stop_reason}", flush=True)
                    break
            else:
                time.sleep(max(0.0, float(cycle_delay)))
                continue
            break

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
    print("=" * 72 + "\n", flush=True)

    payload = ok(
        goal=goal,
        cycles_run=len(cycles),
        cycles=cycles,
        frames=frames,
        frame_path=frames[-1] if frames else None,
        stop_reason=stop_reason,
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
        **{profile.terminal_flag_key: bool(terminal_evidence),
           profile.terminal_evidence_key: terminal_evidence},
        **profile.summarize(counters),
    )
    ctx.artifacts.save_json(profile.json_artifact_name, payload)
    return payload
