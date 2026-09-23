"""
record_play.py - record YOURSELF playing, to teach the agent the route.

WHY THIS IS SEPARATE FROM `console.py play`
-------------------------------------------
This tool never touches the controller. You play with your own pad; it only
WATCHES the capture card and writes down what it sees. That is the whole
point: the agent's own runs are full of failed experiments, whereas your run
is a known-good route. Feeding that route back in turns the agent's job from
"explore blindly" into "verify against a reference".

The capture card is a SINGLE resource. Do not run this at the same time as
`console.py play` - whichever starts second will fail to open the device.

WHAT IT PRODUCES
    artifacts/walkthroughs/<name>/
        play.mp4        the whole session, encoded live by ffmpeg
        frames/         only the KEY frames (scene changes, marker uses)
        route.json      machine-readable beats with timestamps
        ROUTE.md        a prompt fragment to paste into the agent's context

WHILE RECORDING it prints a live beat log, so you can see it noticing your
Magic Marker uses as you make them:

    [00:41.2] MARKER OPENED   saturation 58.3
    [00:43.8] MARKER CLOSED   open for 2.6s -> a drawing was committed
    [00:45.1] SCENE CHANGE    delta 52.9 - camera panned / new area

USAGE
    python record_play.py                       # stop with Ctrl+C
    python record_play.py --name sea-of-sand --minutes 10
    python record_play.py --no-video            # beats only, no mp4
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "core"))
sys.path.insert(0, str(ROOT / "tools"))

from config import Config, load_dotenv_if_present   # noqa: E402
from adapters import HardwareBridge                 # noqa: E402
from walkthrough_video import (                     # noqa: E402
    MARKER_SAT_THRESHOLD, cursor_ring, marker_open,
)

# Ambient noise on this capture path is 2-4; see MARKER_FINDINGS.md.
SCENE_DELTA = 12.0


def fmt(t: float) -> str:
    return f"{int(t // 60):02d}:{t % 60:04.1f}"


class Recorder:
    """Watches the capture card and records the session, without playing."""

    def __init__(self, out_dir: Path, fps: float, want_video: bool,
                 sat_threshold: float, scene_delta: float) -> None:
        self.dir = out_dir
        self.frames_dir = out_dir / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.sat_threshold = sat_threshold
        self.scene_delta = scene_delta
        self.beats: list[dict] = []
        self.encoder: subprocess.Popen | None = None
        self.want_video = want_video
        self.total = 0
        self.started = time.time()

    # -- video -----------------------------------------------------------
    def _open_encoder(self, width: int, height: int) -> None:
        """Pipe raw frames into ffmpeg so the mp4 is written as we go.

        Encoding live avoids holding thousands of frames in RAM or on disk as
        loose JPEGs, and means a Ctrl+C still leaves a playable file.
        """
        if not self.want_video or self.encoder is not None:
            return
        if not shutil.which("ffmpeg"):
            print("  ! ffmpeg not on PATH - recording beats only, no mp4")
            self.want_video = False
            return
        self.encoder = subprocess.Popen(
            ["ffmpeg", "-v", "error", "-y",
             "-f", "rawvideo", "-pix_fmt", "bgr24",
             "-s", f"{width}x{height}", "-r", str(self.fps),
             "-i", "-",
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
             "-pix_fmt", "yuv420p", str(self.dir / "play.mp4")],
            stdin=subprocess.PIPE)

    def _write_video(self, frame) -> None:
        if self.encoder and self.encoder.stdin:
            try:
                self.encoder.stdin.write(frame.tobytes())
            except (BrokenPipeError, OSError):
                self.encoder = None      # encoder died; keep recording beats

    def close_video(self) -> None:
        if self.encoder and self.encoder.stdin:
            try:
                self.encoder.stdin.close()
            except OSError:
                pass
            self.encoder.wait(timeout=60)
            self.encoder = None

    # -- beats -----------------------------------------------------------
    def _add(self, kind: str, t: float, frame, **extra) -> dict:
        """Record a beat and save the frame that proves it."""
        name = f"beat_{len(self.beats) + 1:04d}_{kind}.jpg"
        cv2.imwrite(str(self.frames_dir / name), frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 85])
        beat = {
            "n": len(self.beats) + 1,
            "t": round(t, 2),
            "timestamp": fmt(t),
            "kind": kind,
            "frame": f"frames/{name}",
            **extra,
        }
        self.beats.append(beat)
        return beat

    def run(self, camera, max_seconds: float) -> None:
        prev = None
        prev_open = False
        opened_at = None
        last_scene = -99.0
        next_tick = time.time()
        interval = 1.0 / self.fps

        print("\n  RECORDING - play the game now. Ctrl+C to stop.\n")
        while True:
            now = time.time()
            elapsed = now - self.started
            if max_seconds and elapsed >= max_seconds:
                print(f"\n  reached the {max_seconds / 60:.0f} minute limit")
                return

            # Keep a steady cadence rather than sleeping a fixed amount, so
            # slow frames do not make the video drift out of real time.
            if now < next_tick:
                time.sleep(min(0.01, next_tick - now))
                continue
            next_tick += interval

            frame = camera.grab(allow_blank=True)
            if frame is None:
                continue
            self.total += 1
            h, w = frame.shape[:2]
            self._open_encoder(w, h)
            self._write_video(frame)

            is_open, sat = marker_open(frame, self.sat_threshold)
            delta = None
            if prev is not None:
                delta = float(cv2.absdiff(prev, frame).mean())

            if is_open and not prev_open:
                opened_at = elapsed
                b = self._add("marker_opened", elapsed, frame,
                              saturation=round(sat, 1),
                              cursor_ring_px=cursor_ring(frame))
                print(f"  [{b['timestamp']}] MARKER OPENED   "
                      f"saturation {sat:.1f}")
            elif prev_open and not is_open:
                held = None if opened_at is None else round(
                    elapsed - opened_at, 2)
                b = self._add("marker_closed", elapsed, frame,
                              saturation=round(sat, 1),
                              marker_open_seconds=held)
                print(f"  [{b['timestamp']}] MARKER CLOSED   "
                      f"open for {held}s -> a drawing was committed")
                opened_at = None
            elif (delta is not None and delta >= self.scene_delta
                    and elapsed - last_scene >= 1.5):
                last_scene = elapsed
                note = ("camera panned / new area" if delta > 40
                        else "Max moved or a drawing grew")
                b = self._add("scene_change", elapsed, frame,
                              delta=round(delta, 2))
                print(f"  [{b['timestamp']}] SCENE CHANGE    "
                      f"delta {delta:.1f} - {note}")

            prev, prev_open = frame, is_open

    # -- outputs ---------------------------------------------------------
    def save(self, name: str) -> None:
        draws = [b for b in self.beats if b["kind"] == "marker_closed"]
        dur = time.time() - self.started
        (self.dir / "route.json").write_text(json.dumps({
            "name": name,
            "recorded": datetime.now().isoformat(timespec="seconds"),
            "duration_seconds": round(dur, 1),
            "frames_seen": self.total,
            "sample_fps": self.fps,
            "marker_uses": len(draws),
            "beats": self.beats,
        }, indent=2), encoding="utf-8")

        held = [b["marker_open_seconds"] for b in draws
                if b.get("marker_open_seconds")]
        lines = [
            f"# HUMAN WALKTHROUGH: {name}",
            "",
            f"Recorded {datetime.now():%Y-%m-%d %H:%M}, "
            f"{dur / 60:.1f} minutes, {len(self.beats)} beats, "
            f"{len(draws)} Magic Marker uses.",
            "",
            "## How to use this",
            "",
            "This is a route a HUMAN took to the checkpoint. Treat it as a",
            "guide to what the level requires - not a script. Your own live",
            "frames are always the truth. It tells you WHERE marker draws",
            "were needed and in WHAT ORDER, which is exactly what is hard",
            "to discover by trial and error.",
            "",
            "## Route",
            "",
            "| # | time | event | detail |",
            "|---|------|-------|--------|",
        ]
        for b in self.beats:
            if b["kind"] == "marker_closed":
                detail = (f"drawing committed after "
                          f"{b.get('marker_open_seconds')}s open")
            elif b["kind"] == "marker_opened":
                detail = f"marker opened (saturation {b.get('saturation')})"
            else:
                detail = f"screen delta {b.get('delta')}"
            lines.append(f"| {b['n']} | {b['timestamp']} | {b['kind']} | "
                         f"{detail} |")

        lines += ["", "## Magic Marker summary", ""]
        if draws:
            for i, b in enumerate(draws, 1):
                lines.append(f"{i}. **{b['timestamp']}** - one drawing, "
                             f"marker held open "
                             f"{b.get('marker_open_seconds')}s")
            if held:
                lines += [
                    "",
                    f"Marker-open durations: "
                    f"{', '.join(f'{h}s' for h in held)}",
                    f"Average {sum(held) / len(held):.2f}s per use - a rough "
                    f"guide to how long each draw takes.",
                ]
        else:
            lines.append("_No marker uses detected in this recording._")

        lines += [
            "",
            "## Honesty note",
            "",
            "Controller inputs are NOT recorded - this tool only watches the",
            "screen. Every event above is inferred from pixels: a sharp drop",
            "in saturation means the marker opened (time slows), and a large",
            "frame delta means the camera panned or a structure appeared.",
            "Stick angles and aim times cannot be recovered from video and",
            "must still be solved live, frame by frame.",
            "",
        ]
        (self.dir / "ROUTE.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Record your own gameplay as reference for the agent.")
    ap.add_argument("--name", default=None,
                    help="Folder name, e.g. 'sea-of-sand'.")
    ap.add_argument("--fps", type=float, default=4.0,
                    help="Frames sampled per second (default 4).")
    ap.add_argument("--minutes", type=float, default=0.0,
                    help="Stop automatically after this long (0 = manual).")
    ap.add_argument("--no-video", action="store_true",
                    help="Skip the mp4; record beats and key frames only.")
    ap.add_argument("--sat-threshold", type=float,
                    default=MARKER_SAT_THRESHOLD,
                    help="Saturation below which the marker counts as OPEN. "
                         "Measured: closed frames sit at 70.4-191.0.")
    ap.add_argument("--scene-delta", type=float, default=SCENE_DELTA,
                    help="Delta that counts as a real change (ambient 2-4).")
    args = ap.parse_args()

    name = args.name or f"walk-{datetime.now():%Y%m%d-%H%M%S}"
    out = ROOT / "artifacts" / "walkthroughs" / name

    load_dotenv_if_present(ROOT / ".env")
    settings = Config.load_all(ROOT / "config", {"settings": "settings.yaml"},
                               base=ROOT)["settings"]
    hw = HardwareBridge(settings)

    print("=" * 70)
    print("  RECORD YOUR OWN GAMEPLAY  (the agent will learn this route)")
    print("=" * 70)
    print(f"  Output   : {out}")
    print(f"  Sampling : {args.fps} frames/sec")
    print(f"  Video    : {'no' if args.no_video else 'play.mp4'}")
    stop = "Ctrl+C" if not args.minutes else f"{args.minutes} min or Ctrl+C"
    print(f"  Stop     : {stop}")
    print("  NOTE     : this does NOT touch the controller - you play.")
    print("             Do not run `console.py play` at the same time;")
    print("             the capture card can only be opened once.")
    print("=" * 70)

    rec = Recorder(out, args.fps, not args.no_video,
                   args.sat_threshold, args.scene_delta)
    try:
        camera = hw.capture()
    except Exception as exc:
        print(f"\n  Capture unavailable: {exc}")
        print("  Close RECentral 4 or any other app holding the card.")
        return 3

    try:
        rec.run(camera, args.minutes * 60.0)
    except KeyboardInterrupt:
        print("\n  stopped by user")
    finally:
        # Always finish the mp4 and write the route, even on Ctrl+C, so a
        # long session is never lost.
        rec.close_video()
        rec.save(name)
        hw.close()

    draws = sum(1 for b in rec.beats if b["kind"] == "marker_closed")
    print("\n" + "=" * 70)
    print(f"  frames sampled : {rec.total}")
    print(f"  beats recorded : {len(rec.beats)}")
    print(f"  marker uses    : {draws}")
    print(f"  saved to       : {out}")
    print("=" * 70)
    print("\n  Review ROUTE.md, then hand it to the agent as context.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
