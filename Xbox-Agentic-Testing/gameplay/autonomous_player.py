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
            # Hold LT to open reticle
            print("  -> [MAGIC MARKER] Raising pillar / branch", flush=True)
            pad._send_event("l2", 255, "marker:open")
            time.sleep(0.25)
            aim_dir = direction if direction in ("up", "down", "left", "right") else "up"
            pad.stick("left_stick", direction=aim_dir, duration=min(duration, 0.8), strength=1.0)
            time.sleep(0.1)
            # Hold RT to draw / raise
            pad._send_event("r2", 255, "marker:draw")
            time.sleep(0.45)
            pad._send_event("r2", 0, "marker:draw_done")
            pad._send_event("l2", 0, "marker:close")
            time.sleep(0.3)

        elif action_type == "destroy_drawing":
            pad.press("x", duration=0.2)

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

                try:
                    analysis: GameStateAnalysis = self.runnable.invoke(messages)
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
                        print(f"Action      : {act.action.upper()} dir={act.direction} dur={act.duration:.2f}s", flush=True)
                        self.execute_action(act)

                # 9. Record history
                action_desc = ", ".join(f"{a.action}({a.direction})" for a in analysis.actions) or "edge_jump_grab(right)"
                self.action_history.append({
                    "step": step_idx,
                    "action": action_desc,
                    "observation": f"Delta {delta:.1f}, scene: {analysis.scene_state}",
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
