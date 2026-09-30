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

from pydantic import BaseModel, Field

from registry import ToolContext, fail, ok
from skill_memory import find_skill_impl, list_skills_impl, report_skill_outcome_impl

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


_BLANK_MEAN = 3.0  # mean pixel value below this = black/no-signal frame
_BLANK_RETRIES = 5
_BLANK_WAIT = 1.5
_MAX_BLANK_STREAK = 8


def is_blank_frame(frame: Any) -> bool:
    try:
        return frame is not None and float(frame.mean()) < _BLANK_MEAN
    except Exception:
        return False


def _drain_stale_frames(camera: Any, max_reads: int = 40) -> int:
    """Discard frames queued in the DirectShow buffer while we were idle.

    The driver keeps the frames captured right after the previous read and
    drops newer ones, so an un-drained grab returns a picture from BEFORE the
    last several seconds of input (measured: 'after' frame == 'before' frame).
    Buffered reads return in a few ms; a live read blocks ~1 frame time
    (~16 ms), so stop once two consecutive reads block.
    """
    cap = getattr(camera, "cap", None)
    if cap is None:
        return 0
    live = 0
    reads = 0
    for reads in range(1, max_reads + 1):
        start = time.time()
        try:
            cap.read()
        except Exception:
            break
        live = live + 1 if (time.time() - start) >= 0.008 else 0
        if live >= 2:
            break
    return reads


def grab_nonblank(camera: Any) -> tuple[Any, bool]:
    """Grab a frame; if it is black, wait and retry a few times (loading
    screens / capture hiccups). Returns (frame, still_blank)."""
    _drain_stale_frames(camera)
    frame = camera.grab(allow_blank=True)
    for _ in range(_BLANK_RETRIES):
        if frame is None or not is_blank_frame(frame):
            break
        time.sleep(_BLANK_WAIT)
        frame = camera.grab(allow_blank=True)
    return frame, is_blank_frame(frame)


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


def _current_provider(ctx: ToolContext) -> str | None:
    """The configured LLM provider name, or None if it cannot be resolved -
    never raises, since this only gates an optional latency optimization.
    """
    try:
        from llm import LLMFactory
        return LLMFactory(ctx.settings).default_provider
    except Exception:
        return None


# Anthropic will not actually cache a block under ~1024 tokens (silently
# charged as a normal, uncached write instead) - guarding this avoids
# claiming a speed/cost benefit that a short prompt would never get.
_MIN_CACHEABLE_CHARS = 1024 * 4


def decide(decider: Any, prompt: str, image_b64: str,
          cacheable_prefix: str | None = None,
          provider: str | None = None) -> Any:
    """Send one frame + prompt to the vision model.

    `cacheable_prefix`, if given, is the part of the prompt that is BYTE-
    IDENTICAL every cycle (a game's system_prompt) - split out and marked
    with Anthropic's `cache_control` so repeated cycles reuse the provider's
    cached read of it instead of re-processing the same few thousand tokens
    every single call. Only applied when `provider` is 'anthropic' and the
    prefix clears the real minimum cacheable size; every other provider (or
    a prefix too short to benefit) gets today's single-block prompt,
    unchanged.
    """
    from langchain_core.messages import HumanMessage

    text_blocks: list[dict[str, Any]]
    if (cacheable_prefix and provider == "anthropic"
            and len(cacheable_prefix) >= _MIN_CACHEABLE_CHARS):
        rest = prompt[len(cacheable_prefix):] if prompt.startswith(cacheable_prefix) else prompt
        text_blocks = [
            {"type": "text", "text": cacheable_prefix,
             "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": rest},
        ]
    else:
        text_blocks = [{"type": "text", "text": prompt}]

    message = HumanMessage(content=[
        *text_blocks,
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


def _skill_lookup(ctx: ToolContext, profile: "GameProfile",
                  scene_state: str) -> list[Any] | None:
    """A saved move sequence for this scene_state, reconstructed as real
    `profile.move_model` instances - or None if there is no skill for it yet,
    or its win-rate has not cleared the profile's trust threshold.
    """
    result = find_skill_impl(ctx, profile=profile.key, skill_name=scene_state)
    if not result.get("ok"):
        return None
    entry = result["entry"]
    rate = entry.get("success_rate")
    if rate is None or entry.get("attempts", 0) < profile.skill_min_attempts:
        return None  # not enough evidence yet either way
    if rate < profile.skill_success_threshold:
        return None  # demoted - this skill stopped earning trust
    try:
        return [profile.move_model(**m) for m in entry["moves"]]
    except Exception:
        return None  # a stored shape that no longer matches the model - skip, don't crash


# ===========================================================================
# Self-directed curriculum (opt-in) - the model proposes its OWN next goal
# from the current frame + its own skill library, instead of always chasing
# a single hardcoded default_goal for the whole run. Same idea as Voyager's
# curriculum agent, built as a small addition on top of skill_memory.py
# rather than a new subsystem.
#
# WHAT THIS DOES NOT DO
# -----------------------
# It never silently swaps the run's goal without a visible log line, and it
# never touches profile.default_goal itself - only the in-loop working goal
# for the REST OF THIS RUN. If the proposal call fails for any reason, the
# current goal is kept unchanged - a curriculum that can crash the run over
# a bad LLM response would be worse than no curriculum at all.
# ===========================================================================
class CurriculumProposal(BaseModel):
    """One short self-directed goal proposal for the REST of this run."""

    next_goal: str = Field(
        description="One concrete, concise next goal reachable from the "
                    "CURRENT frame (e.g. 'mine 3 more logs then craft a "
                    "crafting table'). Must be a real next step, not a "
                    "restatement of the current goal.")
    reason: str = Field(
        description="ONE short sentence: why this is the right next step "
                    "given what's visible now and what skills already exist.")
    keep_current_goal: bool = Field(
        default=False,
        description="True if the CURRENT goal is still the right one and "
                    "no change is needed - next_goal/reason are ignored "
                    "when this is true.")


def _propose_next_goal(ctx: ToolContext, profile: "GameProfile",
                       current_goal: str, image_b64: str) -> str:
    """One small structured LLM call: propose the next goal, or keep the
    current one. Never raises - any failure keeps `current_goal` unchanged.
    """
    try:
        skills = list_skills_impl(ctx, profile=profile.key)
        known = ", ".join(
            f"{s['skill_name']} ({s.get('success_rate')})"
            for s in skills.get("skills", [])[:10]) or "none yet"

        decider = build_vision_decider(ctx, CurriculumProposal)
        prompt = (
            f"You are directing what to do next in {profile.key}. "
            f"Current goal: {current_goal}\n"
            f"Skills already proven to work (name and win-rate): {known}\n"
            f"Look at the attached live frame. If the current goal is still "
            f"sensible, set keep_current_goal=true. Otherwise propose ONE "
            f"concrete next goal reachable from what you see now.")
        proposal = decide(decider, prompt, image_b64)
        if proposal.keep_current_goal or not proposal.next_goal.strip():
            return current_goal
        return proposal.next_goal.strip()
    except Exception:
        return current_goal  # a failed proposal keeps the run going, not stuck


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

    # Independent check of a model-declared success state. Return False to
    # reject the claim (the loop keeps playing). Default accepts.
    confirm_success: Callable[[Any, dict[str, Any]], bool] = field(
        default=lambda decision, counters: True)

    # Optional SIMULTANEOUS batch dispatch (move+look+attack in one GIMX
    # call, matching how a real player's hands work at once) - preferred
    # over the one-at-a-time execute_move loop below when a profile
    # provides it. None means "use the old sequential path" so existing
    # profiles (Max) are unaffected unless they opt in.
    execute_moves: Callable[[ToolContext, list[Any]], list[dict[str, Any]]] | None = None

    # Optional SKILL-MEMORY replay (tools/skill_memory.py): when enabled, a
    # scene_state with a high-enough saved win-rate is replayed directly
    # instead of asking the model to re-decide move parameters it has
    # already solved. Off by default - existing profiles are unaffected
    # unless they opt in. forced_moves (safety overrides) is always
    # checked FIRST and always wins over a skill replay.
    use_skill_memory: bool = False
    skill_success_threshold: float = 0.7
    skill_min_attempts: int = 2

    # Optional SELF-DIRECTED CURRICULUM (see CurriculumProposal note above):
    # every `curriculum_interval` cycles, ask the model whether the working
    # goal should change, using the skill library as context. Off by
    # default - existing profiles keep their single fixed default_goal
    # unless they opt in.
    use_curriculum: bool = False
    curriculum_interval: int = 5


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

    provider = _current_provider(ctx)

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
            before, before_blank = grab_nonblank(camera)
            if before is not None and before_blank:
                counters["blank_streak"] = counters.get("blank_streak", 0) + 1
                print(f"  [cycle {cycle}] frame still black after retries "
                      f"(streak {counters['blank_streak']}) - not acting.", flush=True)
                if counters["blank_streak"] >= _MAX_BLANK_STREAK:
                    stop_reason = (f"{_MAX_BLANK_STREAK} consecutive black frames - "
                                   f"capture has no signal.")
                    break
                continue
            counters["blank_streak"] = 0
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

            # --- 1b. self-directed curriculum (opt-in) -----------------------
            # Every `curriculum_interval` cycles, not every cycle - an extra
            # LLM call per cycle would double the round-trip cost for a
            # decision that rarely needs to change that often.
            if (profile.use_curriculum and cycle > 1
                    and (cycle - 1) % max(1, profile.curriculum_interval) == 0):
                proposed = _propose_next_goal(ctx, profile, goal, image_b64)
                if proposed != goal:
                    print(f"  Curriculum: goal changed -> {proposed}", flush=True)
                    goal = proposed

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
                decision = decide(decider, prompt, image_b64,
                                  cacheable_prefix=profile.system_prompt,
                                  provider=provider)
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
            if (decision.scene_state in profile.success_states
                    and profile.confirm_success(decision, counters)):
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
            # forced_moves (safety overrides, e.g. critical-health retreat)
            # always wins - checked first, before any skill replay.
            moves = profile.forced_moves(decision, counters)
            used_skill = False
            if moves is None and profile.use_skill_memory:
                moves = _skill_lookup(ctx, profile, decision.scene_state)
                used_skill = moves is not None
                if used_skill:
                    print(f"  Skill     : replaying saved '{decision.scene_state}' "
                         f"sequence instead of asking the model to re-decide it",
                         flush=True)
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
            after, after_blank = grab_nonblank(camera)
            after_path = (ctx.artifacts.save_frame(after, f"{profile.artifact_prefix}-{cycle:03d}-after")
                          if after is not None else None)
            if after_path:
                frames.append(after_path)

            delta = frame_delta(ctx, before, after)
            if delta is not None:
                deltas.append(delta)
                print(f"  Delta     : {delta:.3f}", flush=True)

            progressed = delta is not None and delta >= 1.0
            if after_blank:
                # Black after-frame (loading/transition): not evidence of
                # being stuck, so do not feed the stuck counters.
                print("  Delta     : (after-frame black - not counted as stuck)",
                      flush=True)
            else:
                profile.update_stuck_counters(counters, moves[:3], progressed)

            if used_skill:
                # The loop's own measured delta IS the evidence - reuse it
                # rather than inventing a second judgment of "did it work."
                report_skill_impl_result = report_skill_outcome_impl(
                    ctx, profile=profile.key, skill_name=decision.scene_state,
                    success=progressed)
                print(f"  Skill     : outcome reported, success_rate now "
                     f"{report_skill_impl_result.get('entry', {}).get('success_rate')}",
                     flush=True)

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
        goal_met=bool(terminal_evidence),
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
