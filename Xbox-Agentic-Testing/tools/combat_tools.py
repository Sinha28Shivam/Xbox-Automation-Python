"""combat_tools.py - single-mob engage/retreat combat prototype.

SMALLEST NEW SCOPE, DELIBERATELY
----------------------------------
This is a PROTOTYPE for exactly one mechanic: approach one hostile mob,
attack it a few times, retreat if it gets too dangerous, stop when it's
gone or the cycle budget runs out. It does NOT attempt multi-mob fights,
ranged combat, or armor/health-bar reading. "Health" here is inferred
only from the mob still being visible/hostile, which is a weak signal and
stated as such in every result.
CORRECTION (2026-09-25): an earlier version of this docstring claimed
Bedrock's UI has no numeric/visual health display at all - that was only
true for CREATIVE mode (both bars are hidden there). In SURVIVAL mode
there IS a real hearts/hunger HUD, and health_tools.read_survival_hud now
reads it via hardware-verified color-mask + contour counting (NOT OCR).
This prototype does not yet CALL that tool to track the PLAYER's own
health during a fight - only the mob's visibility is tracked. Wiring
read_survival_hud into this loop (e.g. to trigger retreat on a real
hearts-dropping signal instead of just "the mob looks dangerous") is a
follow-up, not done yet.

WHY A PROTOTYPE FIRST, NOT A GENERAL COMBAT SYSTEM
----------------------------------------------------
Per the session's own design lesson (the mining stuck-threshold bug): never
assume a pattern/threshold works for a new mechanic without hardware
evidence. Combat is a new mechanic with unknown timing (attack cooldown,
mob aggro range, how much screen-delta an attack swing itself produces vs.
a real hit) - none of that is verified yet. This tool is deliberately
small so those unknowns get discovered on ONE mechanic before any bigger
combat system (multi-mob, base defense) gets built on assumptions.

THE 'attack' ACTION IS UNVERIFIED AS OF THIS TOOL'S INTRODUCTION
-------------------------------------------------------------------
See game_profiles/minecraft_profile.py's MinecraftMove.action field and
execute_minecraft_move's attack branch - it reuses the mine action's
hardware-verified RT/default_press wiring as a short tap, but has not
itself been confirmed against a real mob yet. The very first hardware run
of this tool IS that verification.

THE HONESTY RULE STILL APPLIES
-------------------------------
Nothing here returns `success`. Each cycle records the before/after frame
paths, the measured pixel delta, and the model's own reading of whether the
mob is still visible; the caller decides what that evidence proves. A
"mob_defeated" result means "the model stopped seeing a hostile mob after
attacking it a few times" - a hypothesis, not a confirmed kill (there is no
kill-confirmation UI to read).
"""

from __future__ import annotations

import time
from typing import Any, Literal

from pydantic import BaseModel, Field

from registry import ToolContext, ToolSpec, fail, make_tool, ok

from gameplay_engine import build_vision_decider, decide, encode_frame, format_history, frame_delta
from game_profiles.minecraft_profile import MinecraftMove, NearbyEntity, execute_minecraft_moves
from health_tools import read_survival_hud_impl


class CombatFrameDecision(BaseModel):
    """The model's structured reading of one combat-cycle frame, plus its plan."""

    scene_state: Literal[
        "in_gameplay", "menu_or_prompt", "cutscene_or_loading", "stuck_or_blocked",
    ] = Field(description="Classification of the current screen.")

    threat_visible: bool = Field(
        description="Is a hostile mob visible anywhere in this frame right now?")

    threat_in_range: bool = Field(
        default=False,
        description="Is the hostile mob close enough (roughly under/near the "
                    "crosshair) that an attack swing would plausibly land?")

    same_target_as_before: bool = Field(
        default=True,
        description="If a previous cycle's target kind/category was given in "
                    "the session state below, is the hostile mob you see NOW "
                    "the SAME one (same kind, roughly the same position it "
                    "would be after your last move)? Set FALSE if you now "
                    "think it is a different entity, or if no previous "
                    "target was given yet. This exists because an earlier "
                    "real run re-identified the threat differently EVERY "
                    "cycle (creeper -> llama -> zombie -> drowned) and never "
                    "actually closed in on anything - confirm/deny the same "
                    "target rather than re-classifying from scratch.")

    target_kind: str = Field(
        default="",
        description="What the CURRENT target looks like right now, e.g. "
                    "'zombie', 'creeper' - carried forward next cycle so you "
                    "can confirm/deny it's the same one.")

    nearby_entities: list[NearbyEntity] = Field(
        default_factory=list,
        description="Every mob, structure, or resource worth noting - same "
                    "generic entity list used elsewhere in this framework.")

    reasoning: str = Field(
        description="ONE short sentence: what the threat is doing and your "
                    "engage/retreat/reposition call. Combat needs fraction-"
                    "of-a-second decisions, not a written analysis - no "
                    "step-by-step breakdown.")

    moves: list[MinecraftMove] = Field(
        default_factory=list,
        description="1-3 moves to run this cycle, in order. Prefer ONE move "
                    "when the situation is uncertain so the next frame shows "
                    "its isolated effect.")

    confidence: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Confidence that these moves are the right combat response.")


COMBAT_SYSTEM_PROMPT = """\
You are fighting a SINGLE hostile mob in Minecraft (Xbox Bedrock Edition) on
a real Xbox One, in survival first-person view. You control it through an
emulated controller. The attached image is the LIVE screen right now -
reason only from what you can actually see in it.

GOAL: LOCK ONTO ONE hostile mob and stay on it - do not switch targets
every cycle. If the session state below gives you a previous cycle's
target (kind + rough position), look for THAT SAME mob first and set
`same_target_as_before` accordingly, rather than re-identifying whatever
looks most obviously hostile in the new frame. Move toward your locked
target if it is far, attack it repeatedly once it is close (roughly
centred under the crosshair), and RETREAT (move away) if you take visible
damage flashes or the mob is clearly winning. Stop once the mob is no
longer visible, or the threat is gone.

A REAL HEARTS-DROP SIGNAL MAY BE INJECTED INTO THE SESSION STATE BELOW -
if you see a line saying health dropped, that is measured directly off
the HUD (not your own guess) and OVERRIDES your own judgment: retreat
(move AWAY from the target) this cycle regardless of what you were
planning, even mid-attack.

CONTROLS AVAILABLE TO YOU
  move    : left stick, walk in `direction` (forward=up, back=down)
  look    : right stick, turn the camera in `direction` to face the mob
  attack  : a single short RT tap - swings at whatever is directly under
            the crosshair. Only useful when the mob is CLOSE and CENTRED -
            an attack at a distant or off-centre mob will miss. Each
            attack move is one swing; for repeated swings, return multiple
            attack moves or expect several cycles of attacking.
  wait    : do nothing this cycle, for a screen that is still loading

YOU CANNOT KNOW THE MOB'S NUMERIC HEALTH - only that it keeps being
visible (still a threat) or stops being visible (retreated, out of view,
or defeated - you cannot tell which from the frame alone, and should say
so honestly rather than claiming a kill). This prompt does not currently
tell you the PLAYER's own heart/hunger count either, even though a real
HUD for it exists in survival mode - reason only from what you see in the
image.

SCENE STATE RULES
  in_gameplay        : normal first-person view, hotbar/crosshair visible
  menu_or_prompt      : any overlay, dialog, or prompt is on screen
  cutscene_or_loading : a loading/black/transition screen with no player HUD
  stuck_or_blocked    : repeated attempts are not changing anything at all

Return your reading of the frame and 1-3 moves for this cycle.
"""


def engage_single_mob_impl(
    ctx: ToolContext,
    max_cycles: int = 15,
    # Simultaneous move+look+attack dispatch (execute_minecraft_moves) cuts
    # a cycle's real input time from 2-3 sequential ~250ms subprocess calls
    # down to one, so the old 0.3s pause was partly compensating for launch
    # overhead that no longer exists. Lowered so reactions land faster -
    # still tunable per-call for a slower/safer run.
    cycle_delay: float = 0.15,
    settle_after_move: float = 0.4,
    stuck_limit: int = 5,
) -> dict[str, Any]:
    """Approach, attack, and retreat-if-needed from ONE hostile mob.

    Runs until one of these ends it (bounded, like every other gameplay
    loop in this framework):
      * the model reports no hostile mob visible for 2 consecutive cycles
        (treated as "threat resolved" - see caveat in the returned payload)
      * a REAL hearts-drop (read_survival_hud) crosses a critical floor -
        see _CRITICAL_HEARTS below - a genuine safety stop, not the
        model's own (previously blind) judgment
      * `max_cycles` decision cycles have run
      * `stuck_limit` consecutive cycles show no visual change at all
      * the operator interrupts (Ctrl+C) - partial evidence is still returned
      * the capture device or the model becomes unusable

    FIXES APPLIED after the first live run (see repo memory: player died,
    0 attacks dispatched, target misidentified every cycle):
      1. read_survival_hud is now called every cycle. A measured heart
         drop is injected into the prompt as an OVERRIDE instruction (
         retreat now, regardless of what the model was planning), and a
         drop to/below _CRITICAL_HEARTS forces an immediate hard retreat
         move dispatched directly by this loop - not left to the model.
      2. Target persistence: the previous cycle's `target_kind` is carried
         forward in the prompt, and the model must confirm/deny
         `same_target_as_before` instead of re-classifying the threat from
         scratch every cycle (the exact failure mode that let the player
         die while the model called the mob a creeper, then a llama, then
         a zombie, then a drowned, across 4 cycles).
    """
    try:
        camera = ctx.hardware.capture()
    except Exception as exc:
        return fail(f"Capture unavailable, so combat cannot be vision-guided: {exc}")

    try:
        decider = build_vision_decider(ctx, CombatFrameDecision)
    except Exception as exc:
        return fail(f"Vision model unavailable: {exc}")

    max_cycles = max(1, int(max_cycles))
    cycles: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    attacks_dispatched = 0
    no_threat_streak = 0
    stuck_streak = 0
    dispatched_any = False
    threat_resolved = False
    stop_reason = f"Reached the {max_cycles}-cycle budget."
    started = time.time()

    # Real health tracking (fix #1 - see engage_single_mob_impl docstring).
    # None until the first successful read, so a HUD read failure (e.g.
    # creative mode, or OCR/mask miss) never falsely LOOKS like a heart
    # drop - only two consecutive successful reads are ever compared.
    last_hearts: int | None = None
    min_hearts_observed: int | None = None
    health_retreat_forced = False
    # A drop to/below this many hearts forces an immediate hard retreat
    # dispatched by the LOOP itself, not left to the model's judgment.
    _CRITICAL_HEARTS = 4

    # Target persistence (fix #2). Carried across cycles in the prompt so
    # the model confirms/denies the SAME target instead of re-classifying
    # from scratch every cycle.
    locked_target_kind = ""

    print("\n" + "=" * 72, flush=True)
    print("  ENGAGE SINGLE MOB (combat prototype)", flush=True)
    print("=" * 72, flush=True)
    print(f"  Max cycles : {max_cycles}", flush=True)
    print("  Stop early : press Ctrl+C - evidence so far is kept", flush=True)
    print("=" * 72, flush=True)

    try:
        for cycle in range(1, max_cycles + 1):
            before_frame = camera.grab(allow_blank=True)
            if before_frame is None:
                stop_reason = ("Capture returned no frame - the device may have "
                               "been taken by another application.")
                print(f"  [cycle {cycle}] {stop_reason}", flush=True)
                break

            before_path = ctx.artifacts.save_frame(before_frame, f"combat-{cycle:03d}-before")
            image_b64 = encode_frame(before_frame)
            if image_b64 is None:
                stop_reason = "Could not JPEG-encode the frame for the vision model."
                break

            # Real health check BEFORE deciding this cycle's moves, so a
            # detected drop can be injected as an override instruction the
            # model actually sees - not discovered after the fact.
            hud = read_survival_hud_impl(ctx, frame_path=before_path)
            current_hearts = hud.get("full_hearts") if hud.get("ok") else None
            health_line = ""
            if current_hearts is not None:
                if last_hearts is not None and current_hearts < last_hearts:
                    health_line = (
                        f"REAL HEALTH DROP DETECTED: {last_hearts} -> "
                        f"{current_hearts} hearts (measured off the HUD, not "
                        f"a guess). RETREAT THIS CYCLE regardless of your "
                        f"own plan.\n")
                else:
                    health_line = f"Current health: {current_hearts} hearts.\n"
                last_hearts = current_hearts
                min_hearts_observed = (current_hearts if min_hearts_observed is None
                                       else min(min_hearts_observed, current_hearts))

            target_line = (
                f"Locked target from last cycle: '{locked_target_kind}' - "
                f"look for THIS SAME mob first.\n"
                if locked_target_kind else
                "No target locked yet - identify one and report target_kind.\n")

            prompt = (
                f"{COMBAT_SYSTEM_PROMPT}\n"
                f"# Session state\n"
                f"Cycle: {cycle} of {max_cycles}\n"
                f"Attacks dispatched so far: {attacks_dispatched}\n"
                f"Consecutive no-progress cycles: {stuck_streak}\n"
                f"{health_line}"
                f"{target_line}\n"
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
            print(f"  Scene       : {decision.scene_state}", flush=True)
            print(f"  Threat seen : {decision.threat_visible}"
                  f"  in_range: {decision.threat_in_range}", flush=True)
            print(f"  Target      : {decision.target_kind or '(none)'}"
                  f"  same_as_before: {decision.same_target_as_before}", flush=True)
            if current_hearts is not None:
                print(f"  Health      : {current_hearts} hearts"
                      f"{'  !! CRITICAL' if current_hearts <= _CRITICAL_HEARTS else ''}",
                      flush=True)
            print(f"  Thinking    : {decision.reasoning}", flush=True)

            # Target lock update: only adopt a new target_kind if the model
            # itself says it is NOT the same as before (a genuine switch),
            # or nothing was locked yet - a locked target should not drift
            # just because the model's wording varies cycle to cycle.
            if decision.target_kind and (not locked_target_kind
                                         or not decision.same_target_as_before):
                locked_target_kind = decision.target_kind

            if not decision.threat_visible:
                no_threat_streak += 1
                if no_threat_streak >= 2:
                    threat_resolved = True
                    stop_reason = (
                        f"No hostile mob visible for {no_threat_streak} "
                        f"consecutive cycles - treating the threat as "
                        f"resolved. NOT a confirmed kill: the mob may have "
                        f"fled out of view instead of being defeated, and "
                        f"this UI has no health/kill-confirmation to check.")
                    cycles.append({
                        "cycle": cycle, "frame_before": before_path,
                        "frame_after": before_path, "delta": None,
                        "scene_state": decision.scene_state,
                        "threat_visible": False,
                        "reasoning": decision.reasoning, "moves": [],
                        "confidence": decision.confidence,
                    })
                    print(f"  >> {stop_reason}", flush=True)
                    break
            else:
                no_threat_streak = 0

            # Hard safety override: a critical health drop is NOT left to
            # the model, even though the prompt also told it to retreat -
            # this is the loop's own forced action, matching the fix
            # described in this function's docstring (item 1).
            if current_hearts is not None and current_hearts <= _CRITICAL_HEARTS:
                health_retreat_forced = True
                moves = [MinecraftMove(
                    action="move", direction="down", duration=1.2,
                    purpose=f"FORCED RETREAT: health at {current_hearts} hearts "
                            f"(<= critical floor {_CRITICAL_HEARTS}), overriding "
                            f"the model's own plan.")]
                print(f"  !! Forcing retreat - health at or below critical "
                      f"floor ({current_hearts} <= {_CRITICAL_HEARTS})", flush=True)
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
                        action="look", direction="right", duration=0.5,
                        purpose="Model returned no move; look around for the threat.")]

            # SIMULTANEOUS dispatch (not one-at-a-time): move+look+attack
            # fire together in one GIMX call when the model returns more
            # than one, matching how a real player's hands actually work.
            for move in moves[:3]:
                print(f"  Act         : {move.action} dir={move.direction} "
                      f"dur={move.duration:.2f}s - {move.purpose}", flush=True)
            outcomes = execute_minecraft_moves(ctx, moves[:3])
            dispatched_moves: list[dict[str, Any]] = []
            for move, outcome in zip(moves[:3], outcomes):
                outcome["purpose"] = move.purpose
                dispatched_moves.append(outcome)
                dispatched_any = dispatched_any or bool(outcome.get("dispatched"))
                if move.action == "attack":
                    attacks_dispatched += 1

            time.sleep(max(0.0, float(settle_after_move)))
            after_frame = camera.grab(allow_blank=True)
            after_path = (ctx.artifacts.save_frame(after_frame, f"combat-{cycle:03d}-after")
                          if after_frame is not None else None)

            delta = frame_delta(ctx, before_frame, after_frame)
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
                "threat_visible": decision.threat_visible,
                "threat_in_range": decision.threat_in_range,
                "target_kind": decision.target_kind,
                "same_target_as_before": decision.same_target_as_before,
                "full_hearts": current_hearts,
                "health_retreat_forced_this_cycle": health_retreat_forced,
                "nearby_entities": [e.model_dump() for e in decision.nearby_entities],
                "reasoning": decision.reasoning,
                "moves": dispatched_moves,
                "confidence": decision.confidence,
            })
            health_retreat_forced = False  # reset - only true for the cycle it fired in
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
                    f"change. Either input is not reaching the console or "
                    f"the model is stuck against terrain/the mob. Stopping "
                    f"instead of sending more input into a void.")
                print(f"\n  !! {stop_reason}", flush=True)
                break

            time.sleep(max(0.0, float(cycle_delay)))

    except KeyboardInterrupt:
        stop_reason = ("Interrupted by the operator. Everything observed up to "
                       "this point is kept as evidence.")
        print(f"\n  [stopped] {stop_reason}", flush=True)

    duration = round(time.time() - started, 2)

    print("\n" + "=" * 72, flush=True)
    print(f"  COMBAT PROTOTYPE ENDED after {len(cycles)} cycles in {duration}s", flush=True)
    print(f"  Reason           : {stop_reason}", flush=True)
    print(f"  Attacks dispatched: {attacks_dispatched}", flush=True)
    print("=" * 72 + "\n", flush=True)

    payload = ok(
        cycles_run=len(cycles),
        cycles=cycles,
        stop_reason=stop_reason,
        attacks_dispatched=attacks_dispatched,
        threat_resolved=threat_resolved,
        final_locked_target=locked_target_kind,
        min_hearts_observed=min_hearts_observed,
        duration_seconds=duration,
        dispatched=dispatched_any,
        caveat=(
            "dispatched=true only means GIMX accepted the events, and the "
            "model's reasoning is its own hypothesis - neither is proof. "
            "threat_resolved=true means the mob stopped being VISIBLE, "
            "NOT a confirmed kill - this UI has no health bar or kill "
            "confirmation to check, so a fled/out-of-view mob looks "
            "identical to a defeated one from the evidence available. The "
            "'attack' action's actual hit-registration on a real mob is "
            "UNVERIFIED before this tool's first hardware run - check the "
            "per-cycle deltas for an attack cycle against a real hit "
            "before trusting it works."),
    )
    ctx.artifacts.save_json("combat-cycles.json", payload)
    return payload


def _engage_single_mob(ctx: ToolContext) -> Any:
    def run(max_cycles: int = 15, cycle_delay: float = 0.15,
            stuck_limit: int = 5) -> dict[str, Any]:
        return engage_single_mob_impl(
            ctx, max_cycles=max_cycles, cycle_delay=cycle_delay, stuck_limit=stuck_limit)

    return make_tool(
        run, "engage_single_mob",
        "PROTOTYPE: approach, attack, and retreat-if-needed from ONE "
        "hostile mob. No health-bar reading (Bedrock UI has none) - "
        "'threat_resolved' means the mob stopped being visible, not a "
        "confirmed kill. Bounded like every other gameplay loop: stops at "
        "max_cycles, threat-resolved, a stuck-streak, an operator "
        "interrupt, or a hardware/model failure.")


def provide() -> list[ToolSpec]:
    return [
        ToolSpec(name="engage_single_mob",
                 description=("Prototype single-mob combat: approach, attack, "
                              "retreat-if-needed. No kill confirmation - only "
                              "mob visibility is tracked."),
                 tags=["input", "vision", "game"],
                 factory=_engage_single_mob, mutates_hardware=True),
    ]
