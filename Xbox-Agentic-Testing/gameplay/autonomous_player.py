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
        route_path: str | Path | None = None,
    ):
        self.settings = settings
        self.hardware = hardware or HardwareBridge(settings)
        self.llm_factory = llm_factory or LLMFactory(settings)
        self.game_name = game_name
        self.route_context = self._load_route(route_path)
        # Remembered so the level name can be derived from the route directory
        # name ("sea-of-sand" -> "Sea of Sand") instead of being typed twice.
        self._route_name = Path(route_path).name if route_path else ""
        # The level this session is meant to be playing. Set by
        # launch_from_dashboard, and used by the play loop to recover via
        # SELECT LEVEL instead of resuming the wrong save from CONTINUE.
        self._target_level = self._level_from_route()
        # Vision-guided menus, on by default. Turned off by --no-vision-launch
        # so the OCR keyword path can still be demonstrated or compared.
        self._use_vision_menus = True
        self._navigator: Any | None = None
        # Per-step evidence, persisted to trace.json for the mechanics report.
        self._trace: list[dict[str, Any]] = []
        # Generate the mechanics report at session end. Off via --no-report.
        self._write_report = True
        # Per-screen launch evidence, filled by _vision_navigate_to_level.
        self._launch_trace: dict[str, Any] | None = None
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
        self._requested_actions = []
        self._stuck_cycles = 0
        self._stuck_breaks = 0

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

    # Route context is prepended to every prompt, so it is charged on every
    # cycle. The per-frame notes at the end of ROUTE.md are the least useful
    # part per token, so the file is trimmed rather than sent whole.
    route_char_budget = 6000

    def _load_route(self, route_path: str | Path | None) -> str:
        """Load a human ROUTE.md to give the model the level's running order.

        Accepts either the ROUTE.md itself or a walkthrough session directory
        containing one. Missing or unreadable routes are NOT fatal: playing
        without route context is the previous behaviour, so we warn and carry
        on rather than refusing to start.
        """
        if not route_path:
            return ""
        p = Path(route_path)
        if p.is_dir():
            p = p / "ROUTE.md"
        if not p.is_file():
            print(f"  !! route file not found: {p} - playing without it.",
                  flush=True)
            return ""
        try:
            text = p.read_text(encoding="utf-8").strip()
        except OSError as exc:
            print(f"  !! could not read route {p}: {exc}", flush=True)
            return ""
        if not text:
            return ""

        # Drop the verbose per-frame tail if we are over budget; the route
        # table and "what the level required" list carry the useful signal.
        if len(text) > self.route_char_budget:
            cut = text.find("## Per-frame notes")
            if cut > 0:
                text = text[:cut].rstrip()
        if len(text) > self.route_char_budget:
            text = text[:self.route_char_budget].rstrip() + "\n[...truncated]"

        print(f"  route context loaded from {p} ({len(text)} chars)",
              flush=True)
        return text

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
    # A delta at or below this is ambient noise (idle 2-3, walking 2.7-4.2).
    ambient_delta = 4.5
    # Consecutive ambient cycles before the marker is forced.
    stuck_after = 3
    _stuck_cycles = 0

    def _break_stuck(self, acts: list[GameplayAction],
                     stuck_cycles: int) -> list[GameplayAction]:
        """Force a Magic Marker draw when nothing has moved for a while.

        The per-action veto only catches the SAME macro repeated. Observed
        live: the model cycled move -> running_jump -> push_pull -> jump ->
        edge_jump_grab against the same wall for ten cycles, so no macro ever
        repeated three times and the veto never fired - while a glowing node
        sat right at Max's feet and was described as "no nodes visible".

        Progress is measured by DELTA, not by variety of attempts. After
        `stuck_after` ambient-only cycles, draw on the ground beside Max: in
        this game the marker is nearly always the intended answer, and a
        wrong pillar is cheap because it can be erased.
        """
        if stuck_cycles < self.stuck_after:
            return acts
        if any(a.action in ("magic_marker", "destroy_drawing") for a in acts):
            return acts                      # already trying the marker

        # Alternate the escape route. If a pillar has already been drawn,
        # drawing another one does not help - the problem is more often that
        # Max never gets ON it because every jump goes the wrong way. So on
        # even attempts try climbing LEFT (pillars usually end up behind
        # Max), and on odd attempts draw.
        self._stuck_breaks = getattr(self, "_stuck_breaks", 0) + 1
        if self._stuck_breaks % 2 == 0:
            print(f"  !! {stuck_cycles} stuck cycles - trying a running jump "
                  f"LEFT onto the pillar behind Max.", flush=True)
            return [GameplayAction(action="running_jump", direction="left",
                                   duration=0.9, run_before_jump=0.7,
                                   air_time=0.9)]
        print(f"  !! {stuck_cycles} cycles with no real movement and no "
              f"marker attempt - forcing a pillar draw beside Max.",
              flush=True)
        # Aim just below screen centre: ground level next to Max, where an
        # earth node sits. Riding it up also frees him if he is boxed in.
        return [GameplayAction(action="magic_marker", direction="up",
                               duration=2.0, node_x=0.15, node_y=0.55,
                               aim_time=0.4, settle_after_draw=1.5)]

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
            # Also FLIP the direction. Observed live: Max stood right of his
            # own pillar and kept jumping "right" (i.e. away from it) because
            # right is the level's travel direction, so he landed on empty
            # sand every time. If jumping one way is not working, the target
            # is almost certainly on the other side.
            flipped = "left" if act.direction == "right" else "right"
            print(f"  !! a standing jump {act.direction} twice did nothing - "
                  f"too high, and the target may be behind Max. Running jump "
                  f"{flipped} instead.", flush=True)
            return GameplayAction(action="running_jump", direction=flipped,
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

    # ---- Ink gauge -------------------------------------------------------
    # Holding RT near a node grows a bright ring around the cursor: that is
    # the INK METER. Drawing drains it, and when it is empty the stroke stops
    # regardless of how long the stick is held. So the stroke should not run
    # for a fixed time - it should run until the ink is spent.
    INK_MIN_HSV = (0, 0, 165)      # pale/bright ring pixels
    INK_MAX_HSV = (180, 120, 255)
    # Hardware-measured cursor rest position at 1920x1080 (MARKER_FINDINGS.md).
    CURSOR_REST_X = 967.0
    CURSOR_REST_Y = 534.0
    # Half-width of the gauge search box, as a fraction of the frame. 0.16
    # gives a ~614x346 window at 1080p - wide enough to hold the ring plus
    # aim error, tight enough to keep sand and sky out.
    INK_ROI_HALF = 0.16

    def _cursor_estimate(self, node_x: float, node_y: float, aim_time: float,
                         width: int, height: int) -> tuple[float, float]:
        """Where the cursor should be after aiming, in pixels.

        The cursor opens at its REST position (measured (967,534) at 1080p)
        and travels at ~450 px/s at full deflection. The aim vector is
        normalised the same way `_aim_cursor` normalises it, so this mirrors
        the movement actually dispatched to the pad.
        """
        cx = width * (self.CURSOR_REST_X / 1920.0)
        cy = height * (self.CURSOR_REST_Y / 1080.0)
        hold = max(0.0, min(2.0, float(aim_time)))
        if hold > 0.05 and (abs(node_x) >= 0.05 or abs(node_y) >= 0.05):
            norm = max(abs(float(node_x)), abs(float(node_y))) or 1.0
            travel = self.CURSOR_PX_PER_SEC * hold * (width / 1920.0)
            cx += (float(node_x) / norm) * travel
            cy += (float(node_y) / norm) * travel
        return (max(0.0, min(float(width), cx)),
                max(0.0, min(float(height), cy)))

    def _ink_level(self, frame: Any,
                   centre: tuple[float, float] | None = None) -> float | None:
        """Fraction of the marker ring still filled, as a rough 0..1.

        Measured by the area of the bright, desaturated ring/disc around the
        cursor. Returns None when the marker is not open or the ring cannot
        be found, so callers can fall back to a timed stroke.

        `centre` is the expected cursor position in pixels. The gauge is drawn
        AROUND THE CURSOR, and the cursor moves when we aim - so a fixed
        central ROI loses it. Measured: aiming (+0.7,+0.7) for 0.8s (which the
        prompt recommends for nodes near a screen edge) puts the cursor at
        (1327,894), outside the old fixed box of x 653-1267, y 324-778. The
        gauge then read as "gone" and the stroke was cut back to its minimum,
        which is exactly the "it does not keep growing" symptom.
        """
        if frame is None:
            return None
        try:
            import cv2
            import numpy as np
        except ImportError:
            return None
        # Search ONLY a box around WHERE THE CURSOR IS. Measured: an unbounded
        # bright/pale search fired on 45 of 46 ordinary frames, locking onto
        # sunlit sand and sky instead of the ring, so the ROI restriction is
        # mandatory. But it must TRACK the cursor rather than sit at the rest
        # position, or aiming walks the gauge out of the box.
        height, width = frame.shape[:2]
        cx, cy = centre if centre else (width * 0.5, height * 0.5)
        half_w = width * self.INK_ROI_HALF
        half_h = height * self.INK_ROI_HALF
        x0 = int(max(0, min(width - 2, cx - half_w)))
        x1 = int(max(x0 + 2, min(width, cx + half_w)))
        y0 = int(max(0, min(height - 2, cy - half_h)))
        y1 = int(max(y0 + 2, min(height, cy + half_h)))
        roi = frame[y0:y1, x0:x1]
        if roi.size == 0:
            return None
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array(self.INK_MIN_HSV, dtype=np.uint8),
                           np.array(self.INK_MAX_HSV, dtype=np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                                np.ones((7, 7), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        best = 0.0
        roi_area = float(roi.shape[0] * roi.shape[1])
        for c in contours:
            area = cv2.contourArea(c)
            # The ring is a modest disc, never a huge wash of bright terrain.
            if area < 900 or area > 40000 or area > roi_area * 0.25:
                continue
            x, y, w, h = cv2.boundingRect(c)
            if h == 0 or not 0.7 < w / float(h) < 1.45:
                continue                     # must be roughly circular
            if area / float(w * h) < 0.55:
                continue                     # and solidly filled
            best = max(best, area)
        return best if best > 0 else None

    # Sampling cadence while drawing, and how many flat samples mean "empty".
    ink_poll = 0.25
    ink_flat_samples = 3
    ink_max_hold = 8.0
    # Ring area must fall by at least this many pixels between polls to count
    # as "still draining". Below it, JPEG/compression jitter alone would look
    # like progress and the stroke would never decide the ink was spent.
    ink_drain_step = 150.0

    def _draw_until_empty(self, cap_seconds: float,
                          centre: tuple[float, float] | None = None) -> None:
        """Hold the stroke until the INK runs out, not until a timer expires.

        The stick is ALREADY deflected and A is already held by the caller -
        this only decides WHEN TO STOP. The rule is: keep growing while the
        gauge is still draining, and stop when it has stopped changing for
        `ink_flat_samples` polls in a row (which also covers the ring
        disappearing entirely once spent).

        `cap_seconds` is only a FLOOR for the case where the gauge cannot be
        seen at all. Once the gauge HAS been seen, the ink decides - a short
        requested duration will not cut a stroke short while ink remains,
        because "grow the earth until the ink ends" is the intended
        behaviour. `ink_max_hold` remains as a safety ceiling so a mis-read
        gauge can never hold the pad down forever.
        """
        floor = max(0.3, float(cap_seconds))
        ceiling = max(floor, float(self.ink_max_hold))
        started = time.time()
        try:
            cam = self.hardware.capture()
        except Exception:
            # No camera: fall back to the requested time. Honest degradation -
            # without the gauge there is nothing to close the loop on.
            time.sleep(floor)
            return

        best = None
        flat = 0
        samples = 0
        seen = False               # have we ever located the gauge?
        while True:
            elapsed = time.time() - started
            if elapsed >= ceiling:
                print(f"     ink: hit the {ceiling:.1f}s safety cap "
                      f"(gauge {'tracked' if seen else 'never seen'})",
                      flush=True)
                return
            time.sleep(self.ink_poll)
            elapsed = time.time() - started
            level = self._ink_level(cam.grab(allow_blank=True), centre)

            if level is None:
                if seen:
                    # We were tracking a ring and it has now gone: the tank is
                    # empty. This is a real end-of-ink signal, so it does NOT
                    # wait for the floor.
                    print(f"     ink: gauge emptied after {elapsed:.2f}s "
                          f"({samples} samples)", flush=True)
                    return
                # Never acquired the gauge. Do not guess it is empty - hold
                # for the requested duration and let the caller proceed.
                if elapsed >= floor:
                    print(f"     ink: gauge not visible - held the requested "
                          f"{floor:.2f}s instead", flush=True)
                    return
                continue

            seen = True
            samples += 1
            if best is None or level < best - self.ink_drain_step:
                best, flat = level, 0        # still draining
            else:
                flat += 1
            if flat >= self.ink_flat_samples:
                print(f"     ink: exhausted after {elapsed:.2f}s "
                      f"({samples} samples, last area {level:.0f})",
                      flush=True)
                return

    def _marker_stroke(self, pad: Any, node_x: float, node_y: float,
                       aim_time: float, direction: str,
                       duration: float, settle: float = 1.5,
                       until_empty: bool = True) -> None:
        """OPEN -> AIM -> ANCHOR -> STROKE -> COMMIT -> CLOSE -> SETTLE.

        With `until_empty` the stroke is not a fixed sleep: the ink ring is
        sampled while drawing and the stick is held until the gauge stops
        shrinking (ink exhausted) or `duration` is reached as a safety cap.
        """
        centre = [("lstick x", 0), ("lstick y", 0)]
        stroke = {"up": (0, -32767), "down": (0, 32767),
                  "left": (-32767, 0), "right": (32767, 0)}
        sx, sy = stroke.get(direction, (0, -32767))
        # This is now a MINIMUM hold, not a fixed one: _draw_until_empty
        # keeps drawing while the ink gauge is still draining, up to
        # ink_max_hold. A branch needs to grow heavy enough to fall, and the
        # real limit on any stroke is the ink, not a stopwatch.
        hold = max(0.3, min(6.0, float(duration)))

        # Predict where the cursor will sit after aiming, so the ink gauge is
        # looked for AROUND THE CURSOR rather than at the screen centre.
        cursor = None
        if until_empty:
            try:
                cam = self.hardware.capture()
                probe = cam.grab(allow_blank=True)
                if probe is not None:
                    ph, pw = probe.shape[:2]
                    cursor = self._cursor_estimate(node_x, node_y, aim_time,
                                                   pw, ph)
            except Exception:
                cursor = None            # gauge falls back to frame centre

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
        if until_empty:
            # Tell the gauge where the cursor ended up, so the search box
            # follows the aim instead of sitting at the rest position.
            self._draw_until_empty(hold, cursor)
        else:
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
                        aim_time: float, presses: int = 1,
                        settle: float = 1.5) -> None:
        """Hold RT -> aim at the drawing -> tap X (square) -> release RT.

        WHY THE STICK MUST BE RE-CENTRED BEFORE X
        GIMX axis state is STICKY: an axis holds its last value until it is
        explicitly told otherwise. `_aim_cursor` deflects the stick FULLY and
        returns WITHOUT re-centring it, so any later `_hold` that omits the
        stick leaves it pinned - and the cursor keeps travelling at ~450px/s
        for the whole X sequence.

        Measured on the real trace (aim_time 0.4s, one press): aiming puts the
        cursor on the pillar at ~(1147,714), but the 0.55s of X taps then drag
        it a further ~248px to ~(1394,962) - well off the pillar - so X erased
        empty ground. All three destroys in play-20260907-200600 left the
        pillar's mean colour unchanged (BGR 32.7/69.8/88.7 -> 34.2/71.0/89.5),
        i.e. nothing was ever erased.

        `_marker_draw` never hit this because every one of its holds appends
        `centre`, which re-asserts the stick at 0 and parks the cursor. The
        proven manual gesture does the same: the CONFIRMED WORKING destroy in
        _manual_marker.py runs WITHOUT `--hold-aim`, so its X press sends
        ZERO on the aim axes. Mirroring that here is the fix.

        `settle` stays at drawing's 1.5s because an erase is ANIMATED - the
        pillar crumbles and sinks, and the camera re-frames once it is gone.
        Grabbing the "after" frame too early catches it mid-collapse.
        """
        centre = [("lstick x", 0), ("lstick y", 0)]
        self._hold(pad, [("r2", self.MARKER_RT)] + centre, "destroy:open")
        time.sleep(0.45)
        self._aim_cursor(pad, node_x, node_y, aim_time, "destroy:aim")
        # STOP the cursor on the drawing before tapping X. Without this the
        # stick stays deflected from the aim and the cursor slides off target
        # while X is being pressed. This matches both `_marker_draw` and the
        # hardware-verified manual destroy.
        self._hold(pad, [("r2", self.MARKER_RT)] + centre, "destroy:aim_settle")
        time.sleep(0.20)
        for _ in range(max(1, min(5, int(presses)))):
            self._hold(pad, [("r2", self.MARKER_RT), ("square", 1)] + centre,
                       "destroy:x_down")
            time.sleep(0.25)
            self._hold(pad, [("r2", self.MARKER_RT), ("square", 0)] + centre,
                       "destroy:x_up")
            time.sleep(0.30)
        self._hold(pad, [("r2", 0), ("square", 0)] + centre, "destroy:close")
        time.sleep(max(0.35, float(settle)))

    # ---- Dashboard -> game -> level, then hand over to play() -------------
    # TEMPORARY DEMO PATH. The agentic `run` pipeline is the proper home for
    # launching (it has the planner's prove-focus-before-confirm discipline,
    # the verifier and a real report). This exists so a single command can
    # show the whole journey - dashboard to gameplay - end to end.

    # OCR fragments that mean "we are at Max's LEVEL/CHAPTER PICKER" - actual
    # level names, never the word "chapter" on its own and never "select
    # level", both of which also appear on the MAIN MENU.
    LEVEL_SELECT_SIGNS = ("anotherland", "black rock canyon", "sea of sand",
                          "prologue", "the great fall", "mustacho")

    # The MAIN MENU. Identified by options that exist ONLY there - not by
    # "continue" or "select level", because those are the very items we have
    # to navigate BETWEEN. Two or more of these must be present.
    MAIN_MENU_SIGNS = ("achievements", "options", "help", "extras")

    # "A title/splash screen is waiting for a button."
    #
    # "continue" is DELIBERATELY ABSENT. Observed live: Max's main menu shows
    # CONTINUE as its first, pre-focused item, so treating that word as a
    # title-screen prompt made this helper press A on CONTINUE - which
    # resumed the last save at "Chapter 4-2: The Great Fall" instead of
    # starting Sea of Sand. The main menu is handled by _enter_select_level.
    TITLE_SIGNS = ("press a", "press start", "loading")

    # How far SELECT LEVEL sits below CONTINUE on the main menu.
    SELECT_LEVEL_OFFSET = 1

    def _tool_context(self) -> Any:
        """A ToolContext so the existing game_tools can be reused as-is."""
        from artifacts import ArtifactStore
        from registry import ToolContext

        store = ArtifactStore(
            root=self.settings.resolve_path("paths.artifacts_dir", "./artifacts"),
            run_id=self.session_id,
            frame_format=self.settings.get("runtime.frame_format", "png"),
            enabled=self.settings.get("runtime.save_frames", True),
        )
        return ToolContext(hardware=self.hardware, artifacts=store,
                           settings=self.settings, dry_run=False)

    def _level_from_route(self) -> str:
        """Derive 'Sea of Sand' from artifacts/walkthroughs/sea-of-sand.

        The route directory already names the level, so the demo command does
        not need it spelled out a second time.
        """
        name = getattr(self, "_route_name", "")
        if not name:
            return ""
        return " ".join(part.capitalize() for part in name.split("-"))

    @staticmethod
    def _is_main_menu(text: str) -> bool:
        """Is this Max's MAIN MENU rather than the level picker?

        Requires TWO of the menu-only options. One word is not enough:
        "options" alone could be a pause overlay, and the level picker also
        carries a stray word or two once OCR gets hold of the chapter art.
        """
        hits = sum(1 for sign in AutonomousPlayer.MAIN_MENU_SIGNS
                   if sign in text)
        return hits >= 2

    def _enter_select_level(self, ctx: Any, attempts: int = 5,
                            wait: float = 2.0) -> dict[str, Any]:
        """From the main menu, move to SELECT LEVEL and confirm it.

        CONTINUE is pre-focused and sits directly above SELECT LEVEL. Pressing
        A on the pre-focused item resumes the LAST SAVE - observed live, that
        dropped the run into "Chapter 4-2: The Great Fall" instead of the
        requested level. So we must step DOWN off CONTINUE first.

        After each attempt the screen is re-read: if a level name appears we
        are in the picker and done, and if we are still on the main menu we
        step down again rather than pressing A a second time.
        """
        from game_tools import _observe

        for attempt in range(1, attempts + 1):
            print(f"  [launch] main menu: stepping DOWN off CONTINUE onto "
                  f"SELECT LEVEL (attempt {attempt}/{attempts})", flush=True)
            for _ in range(self.SELECT_LEVEL_OFFSET):
                self.hardware.pad().press("down", duration=0.15)
                time.sleep(0.4)

            self.hardware.pad().press("a", duration=0.2)
            time.sleep(wait)

            obs = _observe(ctx, f"launch-select-level-{attempt}")
            text = str(obs.get("text", "")).lower()
            print(f"  [launch] after SELECT LEVEL: "
                  f"{text[:100] or '(no text)'}", flush=True)

            if any(sign in text for sign in self.LEVEL_SELECT_SIGNS):
                return {"ok": True, "attempt": attempt,
                        "frame_path": obs.get("frame_path"), "text": text}
            if not self._is_main_menu(text):
                # Left the menu but no level name read yet - most likely the
                # picker is still animating in. Let the caller re-check.
                return {"ok": True, "attempt": attempt, "unconfirmed": True,
                        "frame_path": obs.get("frame_path"), "text": text}

        return {"ok": False,
                "error": (f"Could not reach the level picker from the main "
                          f"menu after {attempts} attempts.")}

    def _advance_to_level_select(self, ctx: Any, attempts: int = 10,
                                 wait: float = 3.0) -> dict[str, Any]:
        """Walk logos/splash/title screens and the MAIN MENU to the picker.

        Bounded and OBSERVED rather than a fixed run of blind A presses: Max
        shows a publisher logo, a title card and sometimes a save-slot prompt,
        and how many there are depends on load time. So the screen is read
        between presses, and A is only sent when something is plausibly
        waiting for input.

        The MAIN MENU is treated as its own case. Pressing A there would take
        the pre-focused CONTINUE and resume the last save, which is exactly
        the bug this exists to avoid - we need SELECT LEVEL instead.
        """
        from game_tools import _observe

        for attempt in range(1, attempts + 1):
            obs = _observe(ctx, f"launch-title-{attempt}")
            text = str(obs.get("text", "")).lower()
            print(f"  [launch] screen {attempt}/{attempts}: "
                  f"{text[:100] or '(no text)'}", flush=True)

            # 1. Already in the picker - a real level name is on screen.
            if any(sign in text for sign in self.LEVEL_SELECT_SIGNS):
                return {"ok": True, "attempt": attempt,
                        "frame_path": obs.get("frame_path"), "text": text}

            # 2. The main menu. Navigate to SELECT LEVEL; never press A on
            #    the pre-focused CONTINUE.
            if self._is_main_menu(text):
                entered = self._enter_select_level(ctx)
                if entered.get("ok") and not entered.get("unconfirmed"):
                    return entered
                # Unconfirmed or failed: fall through and re-read the screen
                # on the next pass rather than assuming either way.
                time.sleep(wait)
                continue

            # 3. A blank OCR result is usually a logo or a black transition
            #    frame, which still needs a press to move on.
            if any(sign in text for sign in self.TITLE_SIGNS) or not text.strip():
                self.hardware.pad().press("a", duration=0.2)
            time.sleep(wait)

        return {"ok": False,
                "error": (f"The level picker was not seen after {attempts} "
                          f"screens. The game may still be loading, or it "
                          f"resumed straight into gameplay.")}

    def _resume_via_select_level(self) -> dict[str, Any]:
        """Mid-play recovery: main menu -> SELECT LEVEL -> the target level.

        The play loop can land back on the main menu after a death, a quit or
        a chapter boundary. Pressing A there resumes the last save, so this
        drives the same SELECT LEVEL route the launch step uses.
        """
        from game_tools import select_level_impl

        # Vision first - it can see which row is highlighted, so it will not
        # confirm CONTINUE by accident the way the keyword path could.
        if self._use_vision_menus:
            result = self._vision_navigate_to_level(self._target_level)
            if result.get("ok"):
                time.sleep(10.0)
                return result
            print("  [menu] vision recovery did not finish - falling back to "
                  "the OCR keyword path.", flush=True)

        ctx = self._tool_context()
        entered = self._enter_select_level(ctx)
        if not entered.get("ok"):
            print(f"  !! {entered.get('error')}", flush=True)
            return entered

        selected = select_level_impl(ctx, target_level=self._target_level,
                                     chapter="Chapter 1", max_attempts=8)
        print(f"  [menu] level select: matched={selected.get('matched')} "
              f"-> {selected.get('selected_level')}", flush=True)
        # Loading a level takes a while; without this the next cycle judges a
        # loading screen and burns a decision on it.
        time.sleep(10.0)
        return selected

    def _save_trace(self) -> None:
        """Persist the per-step trace. Never fatal - it is evidence, not state."""
        try:
            (self.session_dir / "trace.json").write_text(
                json.dumps({
                    "session_id": self.session_id,
                    "game_name": self.game_name,
                    "level": self._target_level,
                    "route": getattr(self, "_route_name", ""),
                    "launch": getattr(self, "_launch_trace", None),
                    "steps": self._trace,
                }, indent=2, default=str),
                encoding="utf-8")
        except Exception as exc:
            print(f"  [trace] could not be written: {exc}", flush=True)

    def _menu_navigator(self) -> Any:
        """The vision-guided menu walker, built lazily and reused."""
        from gameplay.menu_navigator import MenuNavigator

        if getattr(self, "_navigator", None) is None:
            self._navigator = MenuNavigator(
                hardware=self.hardware, settings=self.settings,
                artifacts=self._tool_context().artifacts,
                llm_factory=self.llm_factory)
        return self._navigator

    def _vision_navigate_to_level(self, target: str) -> dict[str, Any]:
        """Use the vision navigator to get from wherever we are into `target`.

        The goal is stated in plain English rather than as a screen sequence,
        because the number of logos/title cards varies and the model can see
        which one it is actually looking at.
        """
        goal = (
            f"Start playing the level '{target}' in "
            f"'{self.game_name}'. Get past any publisher logo or title "
            f"screen, then from the game's main menu choose SELECT LEVEL "
            f"(never CONTINUE, which would resume a different save), then "
            f"pick Chapter 1 if a chapter list appears, then choose "
            f"'{target}' and confirm it so the level starts."
        )
        try:
            nav = self._menu_navigator()
            result = nav.navigate_to(goal=goal, target=target,
                                     label="launch-menu")
            # Keep the per-screen launch evidence so the report can show the
            # whole journey - dashboard, logos, main menu, level picker - not
            # just the gameplay that followed it.
            self._launch_trace = {
                "goal": goal,
                "target": target,
                "ok": bool(result.get("ok")),
                "reason": result.get("reason", ""),
                "cycles": result.get("cycles"),
                "target_confirmed": bool(result.get("target_confirmed")),
                "steps": list(getattr(nav, "steps", [])),
            }
            self._save_trace()
            return result
        except Exception as exc:
            # A navigator failure must not kill the demo; the caller falls
            # back to the OCR path.
            print(f"  !! vision navigation unavailable: {exc}", flush=True)
            return {"ok": False, "reason": str(exc)}

    def launch_from_dashboard(self, level: str = "", launch_wait: float = 25.0,
                              settle_after_level: float = 12.0,
                              use_vision: bool = True) -> dict[str, Any]:
        """Dashboard -> locate the game -> launch -> pick the level -> play.

        Reuses tools/game_tools.py rather than re-deriving any of it: the OCR
        title matching, the Store-redirect guard, the "account needs
        attention" bypass and the chapter search are all hardware-verified
        there, and a second implementation here would drift from it.
        """
        from game_tools import _launch_game, select_level_impl

        target = level or self._level_from_route() or "Sea of Sand"
        # Remembered so the play loop's main-menu guard recovers to the SAME
        # level, including when --level overrode the route directory name.
        self._target_level = target
        ctx = self._tool_context()

        print("\n" + "=" * 72, flush=True)
        print(f"  LAUNCH FROM DASHBOARD: {self.game_name}", flush=True)
        print(f"  Target level : {target}", flush=True)
        print("=" * 72, flush=True)

        # 1. Locate the tile and launch it. This helper refuses to press A on
        #    an unidentified tile, and reports a Store/purchase redirect as a
        #    FAILURE rather than as a launched game.
        tool = _launch_game(ctx)
        launched = getattr(tool, "func", tool)(
            game_name=self.game_name, max_tiles=2, launch_wait=launch_wait)
        if not launched.get("ok"):
            print(f"  !! launch failed: {launched.get('error')}", flush=True)
            return {"ok": False, "stage": "launch", **launched}

        if launched.get("already_running"):
            print("  [launch] the game was already on screen - the dashboard "
                  "step was skipped.", flush=True)

        # 2. Walk logos -> title -> main menu -> SELECT LEVEL -> the level.
        #    Vision first: the keyword classifier below cannot tell the main
        #    menu from the level picker (both say "chapter"/"select level"),
        #    and it cannot see WHICH row is highlighted, which is the only
        #    thing that makes pressing A safe.
        if use_vision:
            reached = self._vision_navigate_to_level(target)
            if reached.get("aborted"):
                return {"ok": False, "stage": "menu", **reached}
            if not reached.get("ok"):
                print("  [launch] vision navigation did not finish - falling "
                      "back to the OCR keyword path.", flush=True)
                reached = self._advance_to_level_select(ctx)
            else:
                # Vision drives all the way into the level, so the OCR-based
                # select_level_impl below must not run and re-confirm.
                self.session.metadata["launched_from_dashboard"] = True
                self.session.metadata["launch_target_level"] = target
                self.session.metadata["launch_mode"] = "vision"
                self.session.save(self.session_dir / "session.json")
                return {"ok": True, "target_level": target,
                        "launch": launched, "menu": reached}
        else:
            reached = self._advance_to_level_select(ctx)
        if not reached.get("ok"):
            # NOT fatal. A Quick-Resume console drops straight back into
            # gameplay, which is a perfectly good place to start playing - so
            # say what happened and carry on rather than refusing to run.
            print(f"  !! {reached.get('error')}", flush=True)
            print("  [launch] continuing into gameplay anyway - the play loop "
                  "handles menus and cutscenes itself.", flush=True)
        else:
            # 3. OCR-search the chapters for the level and confirm it.
            selected = select_level_impl(ctx, target_level=target,
                                         chapter="Chapter 1", max_attempts=8)
            print(f"  [launch] level select: matched={selected.get('matched')} "
                  f"-> {selected.get('selected_level')}", flush=True)
            time.sleep(max(0.0, float(settle_after_level)))

        self.session.metadata["launched_from_dashboard"] = True
        self.session.metadata["launch_target_level"] = target
        self.session.save(self.session_dir / "session.json")
        return {"ok": True, "target_level": target, "launch": launched}

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
                # Same settle the draw path gets. The crumble animation needs
                # to finish before the next frame is captured, or the erase
                # is invisible in the evidence.
                settle=getattr(act, "settle_after_draw", 1.5),
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
                    # Count consecutive cycles where nothing really moved, so
                    # varied-but-useless attempts still register as stuck.
                    if delta <= self.ambient_delta:
                        self._stuck_cycles += 1
                    else:
                        self._stuck_cycles = 0
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

                # Route context from a human walkthrough, when supplied. It
                # tells the model the ORDER of obstacles and where a drawing
                # was needed - the part that is expensive to learn by trial
                # and error. It is explicitly subordinate to the live frame.
                route_text = ""
                if self.route_context:
                    route_text = (
                        "\n=== ROUTE CONTEXT FROM A HUMAN WALKTHROUGH ===\n"
                        f"{self.route_context}\n"
                        "=== END ROUTE CONTEXT ===\n"
                        "Use the route for WHAT this level requires and in "
                        "WHAT ORDER. It contains no controller data, so aim "
                        "and stick angles are still yours to solve. If the "
                        "attached frame disagrees with the route, TRUST THE "
                        "FRAME - you may be at a different point than the "
                        "timings suggest.\n"
                    )

                prompt = (
                    f"{MAX_GAMEPLAY_SYSTEM_PROMPT}\n"
                    f"{route_text}"
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
                    # NEVER blind-press A here. Observed live: a "Restart
                    # Level - you will lose all progress" dialog appeared with
                    # "Ok" focused. The model correctly decided to CANCEL, but
                    # this branch pressed A regardless and wiped the level.
                    # Honour whatever the model actually chose; only fall back
                    # to A when it offered nothing.
                    # Max's MAIN MENU is not a harmless prompt. CONTINUE is
                    # pre-focused, so pressing A resumes the LAST SAVE -
                    # observed live at cycles 10-11, that jumped the run into
                    # "Chapter 4-2: The Great Fall" instead of the requested
                    # level, and the route context then described a level we
                    # were no longer in. Route through SELECT LEVEL instead.
                    menu_text = " ".join([
                        str(analysis.interactive_elements or ""),
                        str(analysis.player_location or ""),
                    ]).lower()
                    if self._is_main_menu(menu_text) and self._target_level:
                        print(">> [MAIN MENU] CONTINUE is pre-focused and "
                              "would resume the wrong level - going to "
                              "SELECT LEVEL instead.", flush=True)
                        self._resume_via_select_level()
                        self.action_history.append({
                            "step": step_idx,
                            "action": "select_level",
                            "observation": (
                                f"Main menu detected. Navigated to SELECT "
                                f"LEVEL for '{self._target_level}' rather "
                                f"than pressing A on CONTINUE."),
                        })
                        continue

                    if analysis.actions:
                        for act in analysis.actions[:2]:
                            act = self._veto_repeat(act)
                            print(f">> [PROMPT] {act.action.upper()} "
                                  f"button={act.button or 'default'}",
                                  flush=True)
                            self.execute_action(act)
                            self._recent_actions.append(act.action)
                        chosen = ", ".join(
                            f"{a.action}({a.button or a.direction})"
                            for a in analysis.actions[:2])
                    else:
                        print(">> [PROMPT / CUTSCENE] Advancing screen...",
                              flush=True)
                        self.hardware.pad().press("a", duration=0.2)
                        chosen = "press A to advance"
                    time.sleep(1.5)
                    self.action_history.append({
                        "step": step_idx,
                        "action": chosen,
                        "observation": "Dismissed cutscene / prompt.",
                    })
                    continue

                if analysis.scene_state == "level_complete":
                    print("\n>> [VICTORY] Level completed successfully!", flush=True)
                    self.session.mark_complete()
                    break

                # 8. Execute action sequence
                # Collect what ACTUALLY ran, not what was proposed: both
                # _break_stuck and _veto_repeat can substitute a different
                # macro, and a report built from the proposal would claim a
                # mechanic was exercised when a different one was.
                executed: list[GameplayAction] = []
                if not analysis.actions:
                    # Default: exploratory edge jump grab right
                    print("Action      : Default edge jump grab right", flush=True)
                    fallback = GameplayAction(action="edge_jump_grab",
                                              direction="right", duration=0.8)
                    self.execute_action(fallback)
                    executed.append(fallback)
                else:
                    # Delta-based stuck breaker: catches the case where the
                    # model keeps VARYING its attempts but nothing moves.
                    chosen_acts = self._break_stuck(
                        list(analysis.actions), self._stuck_cycles)
                    for act in chosen_acts:
                        act = self._veto_repeat(act)
                        print(f"Action      : {act.action.upper()} dir={act.direction} dur={act.duration:.2f}s", flush=True)
                        self.execute_action(act)
                        self._recent_actions.append(act.action)
                        executed.append(act)

                # 9. Record history
                action_desc = ", ".join(f"{a.action}({a.direction})" for a in analysis.actions) or "edge_jump_grab(right)"
                # A machine-readable trace, written every cycle. The in-memory
                # action_history below is a one-line summary for the next
                # prompt; it dies with the process and carries no scene or
                # marker detail, so it cannot support a mechanics report.
                # This does, and it survives Ctrl+C via the finally block.
                self._trace.append({
                    "step": step_idx,
                    "actions": [
                        {
                            "action": a.action,
                            "direction": a.direction,
                            "duration": round(float(a.duration), 2),
                            "node_x": round(float(a.node_x), 2),
                            "node_y": round(float(a.node_y), 2),
                            "aim_time": round(float(a.aim_time), 2),
                            "button": a.button,
                        }
                        for a in executed
                    ],
                    "scene_state": analysis.scene_state,
                    "player_detected": bool(analysis.player_detected),
                    "player_location": analysis.player_location,
                    "hazards_observed": analysis.hazards_observed,
                    "interactive_elements": analysis.interactive_elements,
                    "tactical_reasoning": analysis.tactical_reasoning,
                    "delta": None if delta is None else round(float(delta), 2),
                    "delta_verdict": self._delta_verdict(delta),
                    "frame": str(frame_path),
                    # Filled in on the NEXT cycle: _save_live_frame runs before
                    # the action, so this step's after-state IS the next
                    # step's before-frame. Proving a mechanic needs the pair.
                    "frame_after": None,
                })
                if len(self._trace) >= 2:
                    self._trace[-2]["frame_after"] = str(frame_path)
                self._save_trace()

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
            self._save_trace()

            # The mechanics report is generated BEFORE the capture device is
            # released, but it only reads saved frames - so an interrupted
            # session still produces a report of what it did prove.
            if self._write_report:
                try:
                    from gameplay.mechanics_report import build_report
                    written = build_report(self.session_dir,
                                           settings=self.settings,
                                           llm_factory=self.llm_factory)
                    for fmt, path in (written or {}).items():
                        print(f"  report ({fmt}): {path}", flush=True)
                except Exception as exc:
                    print(f"  [report] could not be generated: {exc}",
                          flush=True)

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
