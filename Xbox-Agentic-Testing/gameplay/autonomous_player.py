"""Autonomous Vision-LLM Gameplay Engine.

Controls the Xbox console in real time by observing the live screen,
reasoning about game mechanics with a multimodal LLM, and executing
timed macro actions.
"""
from __future__ import annotations

import base64
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure console doesn't crash on Windows with LLM unicode characters (e.g. arrows)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import cv2
import numpy as np

from config import Config
from core.adapters import HardwareBridge
from core.game_session import GameSession
from core.llm import LLMFactory, structured
from gameplay.gameplay_prompts import (
    GameStateAnalysis,
    GameplayAction,
    MAX_GAMEPLAY_SYSTEM_PROMPT,
)


class AutonomousPlayer:
    """Autonomous gameplay agent that observes, thinks, and plays in real time."""

    def __init__(
        self,
        settings: Config,
        hardware: HardwareBridge | None = None,
        llm_factory: LLMFactory | None = None,
        game_name: str = "Max: The Curse of Brotherhood",
        session_id: str | None = None,
    ):
        self.settings = settings
        self.hardware = hardware or HardwareBridge(settings)
        self.llm_factory = llm_factory or LLMFactory(settings)
        self.game_name = game_name
        self.session_id = session_id or f"play-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        self.root_dir = Path(__file__).resolve().parent.parent
        self.session_dir = self.root_dir / "artifacts" / "gameplay" / self.session_id
        self.frames_dir = self.session_dir / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)

        self.live_image_path = self.settings.resolve_path(
            "paths.live_image", str(self.root_dir / "live.png")
        )

        self.session = GameSession(
            game_name=self.game_name,
            mode="level",
            objective="Autonomous Vision-LLM Gameplay",
            status="running",
        )

        # Build vision-capable model
        self.model = self._build_model()
        self.runnable = structured(self.model, GameStateAnalysis)

        self.action_history: list[dict[str, Any]] = []
        self.death_count = 0
        self.last_frame: np.ndarray | None = None
        # Per-instance, so one session's repeats never leak into another.
        self._recent_actions = []

    def _build_model(self) -> Any:
        # Prefer claude-3-5-sonnet for spatial reasoning if available, or configured model
        provider = self.settings.get("llm.provider", "anthropic")
        configured_model = self.settings.get(
            f"llm.providers.{provider}.params.model", "claude-3-5-sonnet-20241022"
        )
        return self.llm_factory.build(provider=provider, model=configured_model)

    def _save_live_frame(self, frame: np.ndarray, step_index: int) -> Path:
        """Save frame to session archive and atomically overwrite live.png."""
        step_filename = self.frames_dir / f"step-{step_index:04d}.jpg"
        cv2.imwrite(str(step_filename), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])

        # Atomic replace for live.png so viewers never read a partial file
        try:
            live_path = Path(self.live_image_path)
            tmp_path = live_path.parent / f"tmp_{live_path.name}"
            # Write with fast PNG compression
            cv2.imwrite(str(tmp_path), frame, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            if tmp_path.exists():
                os.replace(tmp_path, live_path)
        except Exception as e:
            pass

        return step_filename

    def _encode_frame(self, frame: np.ndarray) -> str:
        """Encode frame to base64 JPEG for model consumption."""
        # Scale to 1280x720 for optimal balance of detail and latency
        h, w = frame.shape[:2]
        if w > 1280:
            scale = 1280 / w
            frame = cv2.resize(frame, (1280, int(h * scale)), interpolation=cv2.INTER_AREA)
        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
        return base64.b64encode(buffer).decode("utf-8")

    # ---- Magic Marker: hardware-verified values from MARKER_FINDINGS.md ---
    # RT is an AXIS spanning 0..32767, not a 0..255 byte. Anything below ~1023
    # leaves the trigger unpressed as far as the console is concerned.
    MARKER_RT = 32767
    # Measured cursor travel at full stick deflection. Strength matters far
    # more than time: ~450 px/s at 1.0 but only ~17 px/s at 0.4.
    CURSOR_PX_PER_SEC = 450

    # A vision call that never returns would hang the session silently.
    llm_timeout = 90.0

    def _invoke_with_timeout(self, messages: Any, timeout: float = 90.0) -> Any:
        """Run the model on a worker thread so a stuck call cannot hang us.

        The thread is daemonic: if the provider never answers, we abandon the
        call and continue with the next cycle rather than freezing forever.
        """
        import threading

        box: dict[str, Any] = {}

        def work() -> None:
            try:
                box["value"] = self.runnable.invoke(messages)
            except Exception as exc:            # re-raised on the caller side
                box["error"] = exc

        worker = threading.Thread(target=work, daemon=True)
        worker.start()
        worker.join(max(5.0, float(timeout)))
        if worker.is_alive():
            raise TimeoutError(f"vision model exceeded {timeout:.0f}s")
        if "error" in box:
            raise box["error"]
        return box["value"]

    # Rolling record of dispatched macros, for breaking repeat loops.
    _recent_actions: list[str] = []

    def _veto_repeat(self, act: GameplayAction) -> GameplayAction:
        """Break out of a macro the model keeps repeating without effect.

        Observed live: the model read the blue 'X' badge on its own pillar as
        "destroy me" and issued destroy_drawing FOUR cycles running, each with
        an ambient-only delta, while the pillar stayed standing. The pillar was
        the solution - it needed CLIMBING. Prompting alone did not stop this,
        so the loop now forces a different macro after two identical tries.
        """
        # Track what the MODEL ASKED FOR, not what we substituted. Recording
        # the substitute would reset the streak and allow an endless
        # destroy/destroy/climb/destroy... cycle.
        self._requested_actions = getattr(self, "_requested_actions", [])
        self._requested_actions.append(act.action)
        recent = self._requested_actions[-3:]
        if len(recent) < 3 or not all(a == act.action for a in recent):
            return act

        if act.action == "destroy_drawing":
            print("  !! destroy_drawing twice already had no effect - the "
                  "pillar is the SOLUTION, not the obstacle. Climbing it "
                  "instead.", flush=True)
            return GameplayAction(action="edge_jump_grab",
                                  direction=act.direction if act.direction in
                                  ("left", "right") else "right",
                                  duration=0.9, run_before_jump=0.7,
                                  air_time=0.9)
        if act.action == "jump":
            print("  !! a standing jump twice did nothing - too high. "
                  "Running jump from the edge instead.", flush=True)
            return GameplayAction(action="running_jump",
                                  direction=act.direction if act.direction in
                                  ("left", "right") else "right",
                                  duration=0.9, run_before_jump=0.7,
                                  air_time=0.9)
        if act.action == "push_pull":
            print("  !! push_pull twice did nothing - that object will not "
                  "move. Looking for a marker node instead.", flush=True)
            return GameplayAction(action="magic_marker", direction="up",
                                  duration=1.8, node_y=0.6, aim_time=0.4,
                                  settle_after_draw=1.5)
        return act

    @staticmethod
    def _delta_verdict(delta: float | None) -> str:
        """Translate a raw pixel delta into what it actually means here.

        Measured on this rig: an IDLE screen (swaying foliage, drifting dust)
        already reads 2-3, and Max walking only reaches 2.7-4.2. The two
        ranges OVERLAP, so a bare "delta 3.6" was being read by the model as
        "steady progress" when nothing had moved at all - which is why it kept
        pushing rocks for eight cycles instead of drawing.
        """
        if delta is None:
            return "not measured"
        if delta < 2.0:
            return "NOTHING CHANGED - that attempt did nothing"
        if delta < 4.5:
            return ("AMBIENT ONLY - this is the same as an idle screen, so "
                    "Max most likely did NOT move; try a different approach")
        if delta < 12.0:
            return "something really moved"
        return "big change - scene transition, draw, or death"

    @staticmethod
    def _hold(pad: Any, events: list[tuple[str, int]], label: str) -> bool:
        """Assert several controller states in ONE gimx call.

        Sending them one at a time lets a hold lapse between calls, which
        silently ruins the gesture: RT has to stay down for the whole stroke
        or the marker closes and the ink is discarded.
        """
        sender = getattr(pad, "_send_events", None)
        if callable(sender):
            return bool(sender(events, label))
        okay = True
        for control, value in events:
            okay = bool(pad._send_event(control, value, label)) and okay
        return okay

    def _aim_cursor(self, pad: Any, node_x: float, node_y: float,
                    aim_time: float, label: str) -> bool:
        """Steer the marker cursor onto a node at FULL deflection."""
        aim_hold = max(0.0, min(2.0, float(aim_time)))
        if aim_hold <= 0.05 or (abs(node_x) < 0.05 and abs(node_y) < 0.05):
            return False
        norm = max(abs(float(node_x)), abs(float(node_y))) or 1.0
        nx = int(round(max(-1.0, min(1.0, node_x / norm)) * 32767))
        ny = int(round(max(-1.0, min(1.0, node_y / norm)) * 32767))
        print(f"     aim cursor ({node_x:+.2f},{node_y:+.2f}) for "
              f"{aim_hold:.2f}s (~{aim_hold * self.CURSOR_PX_PER_SEC:.0f}px)",
              flush=True)
        self._hold(pad, [("r2", self.MARKER_RT),
                         ("lstick x", nx), ("lstick y", ny)], label)
        time.sleep(aim_hold)
        return True

    def _marker_stroke(self, pad: Any, node_x: float, node_y: float,
                       aim_time: float, direction: str,
                       duration: float, settle: float = 1.5) -> None:
        """OPEN -> AIM -> ANCHOR -> STROKE -> COMMIT -> CLOSE -> SETTLE."""
        centre = [("lstick x", 0), ("lstick y", 0)]
        stroke = {"up": (0, -32767), "down": (0, 32767),
                  "left": (-32767, 0), "right": (32767, 0)}
        sx, sy = stroke.get(direction, (0, -32767))
        # Up to 6s: a tree branch has to grow far enough to become heavy and
        # then physically FALL, which takes much longer than raising a short
        # earth pillar. Cutting the stroke off early leaves a stub that never
        # drops, so the route stays blocked.
        hold = max(0.3, min(6.0, float(duration)))

        self._hold(pad, [("r2", self.MARKER_RT)] + centre, "marker:open")
        time.sleep(0.45)

        if self._aim_cursor(pad, node_x, node_y, aim_time, "marker:aim"):
            self._hold(pad, [("r2", self.MARKER_RT)] + centre,
                       "marker:aim_settle")
            time.sleep(0.20)

        # A is a BUTTON: 1/0, never 255.
        self._hold(pad, [("r2", self.MARKER_RT), ("cross", 1)] + centre,
                   "marker:anchor")
        time.sleep(0.35)
        self._hold(pad, [("r2", self.MARKER_RT), ("cross", 1),
                         ("lstick x", sx), ("lstick y", sy)], "marker:stroke")
        time.sleep(hold)
        self._hold(pad, [("r2", self.MARKER_RT), ("cross", 1)] + centre,
                   "marker:stroke_end")
        time.sleep(0.15)
        self._hold(pad, [("r2", self.MARKER_RT), ("cross", 0)] + centre,
                   "marker:commit")
        time.sleep(0.25)
        self._hold(pad, [("r2", 0), ("cross", 0)] + centre, "marker:close")
        # Let the drawing SETTLE before the next frame is judged. A grown
        # branch keeps moving after the stroke ends - it bends, snaps and
        # falls under its own weight - and a pillar finishes rising. Grabbing
        # a frame immediately shows the mid-animation state, so the next
        # cycle would react to a branch that has not landed yet.
        time.sleep(max(0.35, float(settle)))

    def _marker_destroy(self, pad: Any, node_x: float, node_y: float,
                        aim_time: float, presses: int = 1) -> None:
        """Hold RT -> aim at the drawing -> tap X (square) -> release RT."""
        centre = [("lstick x", 0), ("lstick y", 0)]
        self._hold(pad, [("r2", self.MARKER_RT)] + centre, "destroy:open")
        time.sleep(0.45)
        self._aim_cursor(pad, node_x, node_y, aim_time, "destroy:aim")
        # Keep the aim applied: re-centring can let the cursor drift off the
        # drawing before X registers.
        for _ in range(max(1, min(5, int(presses)))):
            self._hold(pad, [("r2", self.MARKER_RT), ("square", 1)],
                       "destroy:x_down")
            time.sleep(0.25)
            self._hold(pad, [("r2", self.MARKER_RT), ("square", 0)],
                       "destroy:x_up")
            time.sleep(0.30)
        self._hold(pad, [("r2", 0), ("square", 0)] + centre, "destroy:close")
        time.sleep(0.35)

    def execute_action(self, act: GameplayAction) -> None:
        """Dispatch a high-level gameplay action through ConsolePad."""
        pad = self.hardware.pad()
        action_type = act.action
        direction = act.direction
        duration = float(act.duration)

        if action_type == "move":
            if direction in ("left", "right", "up", "down"):
                pad.stick("left_stick", direction=direction, duration=duration, strength=1.0)
            else:
                pad.stick("left_stick", direction="right", duration=duration, strength=1.0)

        elif action_type == "jump":
            pad.press("a", duration=min(duration, 0.25))

        elif action_type == "running_jump":
            run_time = float(act.run_before_jump or 0.35)
            air_time = float(act.air_time or 0.65)
            stick_dir = direction if direction in ("left", "right") else "right"
            stick_val = 32767 if stick_dir == "right" else -32768

            # 1. Start sprinting forward
            pad._send_event("lstick x", stick_val, f"run:{stick_dir}")
            time.sleep(run_time)
            # 2. Jump while holding sprint
            pad.press("a", duration=0.20)
            # 3. Carry momentum through the air
            time.sleep(air_time)
            # 4. Release stick
            pad._send_event("lstick x", 0, "run:release")
            time.sleep(0.2)

        elif action_type == "edge_jump_grab":
            # Precision edge sprint, high jump, reaching forward-up to grip ledge/rope, and pulling up
            run_time = float(act.run_before_jump or 0.45)
            air_time = float(act.air_time or 0.75)
            stick_dir = direction if direction in ("left", "right") else "right"
            stick_x = 32767 if stick_dir == "right" else -32768

            print(f"  -> [EDGE-GRAB] Sprint {stick_dir} ({run_time:.2f}s) -> Jump & Reach -> Climb Ledge", flush=True)
            # 1. Sprint to platform brink
            pad._send_event("lstick x", stick_x, f"run:{stick_dir}")
            time.sleep(run_time)
            # 2. Maximum height jump
            pad.press("a", duration=0.22)
            # 3. Reach forward & UP in flight to catch edge/rope
            pad._send_event("lstick y", -24000, "grip:reach_up")
            time.sleep(air_time)
            # 4. Hoist up onto the ledge
            pad._send_event("lstick y", -32768, "climb:pull_up")
            pad.press("a", duration=0.15)
            time.sleep(0.35)
            # 5. Center sticks
            pad._send_event("lstick x", 0, "release:x")
            pad._send_event("lstick y", 0, "release:y")
            time.sleep(0.15)

        elif action_type == "climb_or_pull_up":
            print("  -> [CLIMB] Pulling up onto ledge/vine", flush=True)
            pad._send_event("lstick y", -32768, "climb:up")
            pad.press("a", duration=0.22)
            pad._send_event("lstick x", 32767, "climb:forward")
            time.sleep(min(duration, 0.7))
            pad._send_event("lstick y", 0, "release:y")
            pad._send_event("lstick x", 0, "release:x")
            time.sleep(0.15)

        elif action_type == "swing_and_jump":
            print("  -> [SWING] Building momentum and leaping from vine/rope", flush=True)
            # Swing back
            pad._send_event("lstick x", -32768, "swing:back")
            time.sleep(0.4)
            # Swing forward
            pad._send_event("lstick x", 32767, "swing:forward")
            time.sleep(0.5)
            # Jump forward
            pad.press("a", duration=0.22)
            time.sleep(0.6)
            pad._send_event("lstick x", 0, "release:x")
            time.sleep(0.2)

        elif action_type == "push_pull":
            stick_dir = direction if direction in ("left", "right") else "right"
            stick_x = 32767 if stick_dir == "right" else -32768
            print(f"  -> [PUSH/PULL] Gripping with B and pushing {stick_dir}", flush=True)
            pad._send_event("circle", 1, "grab:hold")
            pad._send_event("lstick x", stick_x, f"push:{stick_dir}")
            time.sleep(duration)
            pad._send_event("lstick x", 0, "push:release")
            pad._send_event("circle", 0, "grab:release")
            time.sleep(0.2)

        elif action_type == "magic_marker":
            # HARDWARE-VERIFIED sequence - see MARKER_FINDINGS.md.
            # The previous version was wrong in four separate ways:
            #   * it opened the marker with LT ("l2"); the marker is RT ("r2")
            #   * it used 255, but the trigger axis spans 0..32767, so 255 is
            #     ~0.8% of a pull and the console sees an UNPRESSED trigger
            #   * it treated RT as the "draw" button; drawing is A ("cross")
            #   * it aimed and drew in separate calls, so no hold was ever
            #     asserted at the same time as another
            print("  -> [MAGIC MARKER] Raising pillar / branch", flush=True)
            self._marker_stroke(
                pad,
                node_x=getattr(act, "node_x", 0.0),
                node_y=getattr(act, "node_y", 0.0),
                aim_time=getattr(act, "aim_time", 0.4),
                direction=direction,
                duration=duration,
                settle=getattr(act, "settle_after_draw", 1.5),
            )

        elif action_type == "destroy_drawing":
            # A bare X press cannot erase: X only works while RT holds the
            # marker OPEN, and the cursor must be steered onto the drawing
            # first (there is no auto-snap when erasing).
            print("  -> [DESTROY] Erasing a drawing", flush=True)
            self._marker_destroy(
                pad,
                node_x=getattr(act, "node_x", 0.0),
                node_y=getattr(act, "node_y", 0.0),
                aim_time=getattr(act, "aim_time", 0.4),
            )

        elif action_type == "interact":
            pad.press("b", duration=min(duration, 0.3))

        elif action_type == "press_button":
            btn = act.button or "a"
            pad.press(btn, duration=min(duration, 0.2))

        elif action_type == "wait":
            time.sleep(duration)

    def play(self, max_steps: int = 50, cycle_delay: float = 0.4) -> dict[str, Any]:
        """Main autonomous gameplay loop."""
        print("\n" + "=" * 72, flush=True)
        print(f"  AUTONOMOUS AI GAMEPLAY: {self.game_name.upper()}", flush=True)
        print("=" * 72, flush=True)
        print(f"  Session ID : {self.session_id}", flush=True)
        print(f"  Live Feed  : {self.live_image_path}", flush=True)
        print(f"  Max Steps  : {max_steps}", flush=True)
        print("=" * 72 + "\n", flush=True)

        cam = self.hardware.capture()
        consecutive_static = 0
        started_time = time.time()

        try:
            for step_idx in range(1, max_steps + 1):
                # 1. Capture current live frame
                frame = cam.grab(allow_blank=True)
                if frame is None:
                    print(f"[{step_idx:03d}] Warning: No frame returned from capture device.", flush=True)
                    time.sleep(1.0)
                    continue

                # 2. Save frame & update live.png
                frame_path = self._save_live_frame(frame, step_idx)

                # 3. Calculate screen delta from previous step
                delta = 0.0
                if self.last_frame is not None:
                    delta = float(cv2.absdiff(self.last_frame, frame).mean())
                    if delta < 1.0:
                        consecutive_static += 1
                    else:
                        consecutive_static = 0
                self.last_frame = frame.copy()

                # 4. Format prompt with recent action history
                history_text = ""
                if self.action_history:
                    recent = self.action_history[-3:]
                    history_lines = [
                        f"- Step {h['step']}: {h['action']} -> Result: {h['observation']}"
                        for h in recent
                    ]
                    history_text = "\nRecent Action History:\n" + "\n".join(history_lines) + "\n"

                prompt = (
                    f"{MAX_GAMEPLAY_SYSTEM_PROMPT}\n"
                    f"Current Step: {step_idx}/{max_steps}\n"
                    f"Deaths so far: {self.death_count}\n"
                    f"Consecutive Low Progress Steps: {consecutive_static}\n"
                    f"Last Action Screen Delta: {delta:.2f}\n"
                    f"{history_text}\n"
                    f"Analyze the attached frame and return your tactical assessment and actions."
                )

                # 5. Invoke multimodal LLM
                b64_image = self._encode_frame(frame)
                image_block = [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}}]

                from langchain_core.messages import HumanMessage
                messages = [HumanMessage(content=[{"type": "text", "text": prompt}, *image_block])]

                # A hung provider call would otherwise freeze the whole
                # session with no output at all, holding the pad and the
                # capture card. Bound it and move on to the next cycle.
                print(f"[{step_idx:03d}] thinking ...", flush=True)
                try:
                    analysis: GameStateAnalysis = self._invoke_with_timeout(
                        messages, timeout=self.llm_timeout)
                except TimeoutError:
                    print(f"[{step_idx:03d}] Vision LLM did not answer within "
                          f"{self.llm_timeout:.0f}s - skipping this cycle.",
                          flush=True)
                    continue
                except Exception as exc:
                    print(f"[{step_idx:03d}] Vision LLM error: {exc}. Retrying in 1s...", flush=True)
                    time.sleep(1.0)
                    continue

                # 6. Log thinking & plan
                print(f"\n--- [CYCLE {step_idx:03d}] ---", flush=True)
                print(f"Scene       : {analysis.scene_state.upper()}", flush=True)
                print(f"Player      : {'Visible' if analysis.player_detected else 'Searching'} ({analysis.player_location})", flush=True)
                print(f"Hazards     : {analysis.hazards_observed}", flush=True)
                print(f"Interactives: {analysis.interactive_elements}", flush=True)
                print(f"Thinking    : {analysis.tactical_reasoning}", flush=True)
                if analysis.stuck_recovery_tactic:
                    print(f"Recovery    : {analysis.stuck_recovery_tactic}", flush=True)

                # 7. Handle special scene states
                if analysis.scene_state == "death_or_respawn":
                    self.death_count += 1
                    print(">> [DEATH DETECTED] Respawning from checkpoint...", flush=True)
                    self.hardware.pad().press("a", duration=0.2)
                    time.sleep(2.0)
                    self.action_history.append({
                        "step": step_idx,
                        "action": "respawn",
                        "observation": "Died in previous action. Respawned at checkpoint.",
                    })
                    continue

                if analysis.scene_state in ("cutscene_or_loading", "menu_or_prompt"):
                    print(">> [PROMPT / CUTSCENE] Advancing screen...", flush=True)
                    self.hardware.pad().press("a", duration=0.2)
                    time.sleep(1.5)
                    self.action_history.append({
                        "step": step_idx,
                        "action": "press A to advance",
                        "observation": "Dismissed cutscene / prompt.",
                    })
                    continue

                if analysis.scene_state == "level_complete":
                    print("\n>> [VICTORY] Level completed successfully!", flush=True)
                    self.session.mark_complete()
                    break

                # 8. Execute action sequence
                if not analysis.actions:
                    # Default: exploratory edge jump grab right
                    print("Action      : Default edge jump grab right", flush=True)
                    self.execute_action(GameplayAction(action="edge_jump_grab", direction="right", duration=0.8))
                else:
                    for act in analysis.actions:
                        act = self._veto_repeat(act)
                        print(f"Action      : {act.action.upper()} dir={act.direction} dur={act.duration:.2f}s", flush=True)
                        self.execute_action(act)
                        self._recent_actions.append(act.action)

                # 9. Record history
                action_desc = ", ".join(f"{a.action}({a.direction})" for a in analysis.actions) or "edge_jump_grab(right)"
                self.action_history.append({
                    "step": step_idx,
                    "action": action_desc,
                    # Raw deltas were being misread as "steady progress": on
                    # this rig an IDLE screen already measures 2-3 and walking
                    # only reaches 2.7-4.2, so 3.5 means Max probably did NOT
                    # move. Spell that out instead of leaving a bare number.
                    "observation": (
                        f"Delta {delta:.1f} ({self._delta_verdict(delta)}), "
                        f"scene: {analysis.scene_state}"),
                })

                # 10. Checkpoint session
                self.session.checkpoint_now(
                    checkpoint=f"step-{step_idx}",
                    frame=str(frame_path),
                    verdict="running",
                )
                self.session.save(self.session_dir / "session.json")

                time.sleep(cycle_delay)

        except KeyboardInterrupt:
            print("\n[STOPPED] Autonomous gameplay interrupted by user.")
        finally:
            self.session.metadata["duration_seconds"] = round(time.time() - started_time, 2)
            self.session.metadata["total_steps"] = len(self.action_history)
            self.session.metadata["deaths"] = self.death_count
            self.session.save(self.session_dir / "session.json")
            self.hardware.close()

        print("\n" + "=" * 72)
        print("  GAMEPLAY SESSION COMPLETE")
        print(f"  Total Steps : {len(self.action_history)}")
        print(f"  Deaths      : {self.death_count}")
        print(f"  Saved To    : {self.session_dir}")
        print("=" * 72 + "\n")

        return {
            "session_id": self.session_id,
            "steps": len(self.action_history),
            "deaths": self.death_count,
            "session_dir": str(self.session_dir),
        }
