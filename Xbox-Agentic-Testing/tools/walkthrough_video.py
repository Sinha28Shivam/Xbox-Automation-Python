"""
walkthrough_video.py - turn a human playthrough video into a route plan.

SUPERSEDED FOR MARKER DETECTION - USE tools/route_review.py
----------------------------------------------------------
The saturation test below is WRONG on real captures. Measured on the first
real recording (artifacts/walkthroughs/sea-of-sand, 51s): mean saturation
stayed in 78.4-155.5 for the whole clip and never crossed the 62 threshold,
yet the marker was opened at least 8 times (verified by eye at t=8.0s and
t=19.5s, where a freshly drawn pillar carries the erasable X badge). Opening
the marker does NOT globally desaturate this capture path, so this tool
reported "0 Magic Marker uses" for a run full of them.

A pale-disc replacement was tried and rejected: hazy sky scored 82,177px and
pale debris 17,297px, while the real cursor was only ~10,300px - so no size
or roundness threshold separates them. See MARKER_FINDINGS.md.

The scene_change/delta logic here is still sound. For marker events, use
route_review.py, which shows the frames to the vision model instead.

WHAT THIS DOES
--------------
Given a video of someone playing to a checkpoint, it produces:

  1. KEY MOMENTS  - the frames where the screen genuinely changed (a pillar
                    grew, the camera panned, a new area loaded), rather than
                    one frame every N seconds.
  2. MARKER EVENTS- the moments the Magic Marker was OPEN, detected from the
                    time-slow desaturation plus the cursor ring.
  3. A ROUTE FILE - an ordered list of beats with timestamps and the frame
                    paths, ready to be pasted into the gameplay prompt as
                    "here is the route a human took".

WHAT IT DOES NOT DO
-------------------
It cannot read the human's controller. Button presses are INFERRED from what
changed on screen, so every inferred action is written as a hypothesis with
the evidence beside it - never as fact. A pillar appearing proves a draw
happened; it does not prove which stick angle was used.

USAGE
    python tools/walkthrough_video.py run.mp4 --out artifacts/route
    python tools/walkthrough_video.py run.mp4 --fps 2 --scene-delta 12
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

# Ambient noise on this capture path sits at 2-4; a real scene change is
# an order of magnitude bigger. See MARKER_FINDINGS.md.
AMBIENT_DELTA = 4.5


def probe(video: Path) -> dict:
    """Duration/size via ffprobe, so we can report progress honestly."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height,duration,nb_frames",
             "-of", "json", str(video)],
            capture_output=True, text=True, timeout=60)
        return json.loads(out.stdout or "{}").get("streams", [{}])[0]
    except Exception:
        return {}


def extract(video: Path, out_dir: Path, fps: float) -> list[Path]:
    """Sample the video at `fps` using ffmpeg (handles any codec)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("f_*.jpg"):
        old.unlink()
    subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(video),
         "-vf", f"fps={fps},scale=1280:-2", "-q:v", "4",
         str(out_dir / "f_%05d.jpg")],
        check=True, timeout=1800)
    return sorted(out_dir.glob("f_*.jpg"))


# Whole-frame mean saturation below this is treated as "marker open".
# CALIBRATION EVIDENCE: across 158 real marker-CLOSED frames the mean
# saturation ranged 70.4 - 191.0 (mean 131.7), so nothing normal falls below
# ~70. On the cursor region saturation was measured dropping 123 -> 53 when
# the marker opened. 62 sits below every observed closed frame while staying
# above that measured open value - but the margin to 70.4 is thin, so this is
# deliberately tunable with --sat-threshold.
MARKER_SAT_THRESHOLD = 62.0


def marker_open(img: np.ndarray,
                threshold: float = MARKER_SAT_THRESHOLD) -> tuple[bool, float]:
    """Is the Magic Marker open in this frame?

    When the marker opens, time slows and the palette DESATURATES sharply.
    That global shift is far more reliable than hunting for the small cursor
    ring, which is easily confused with sunlit sand.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sat = float(hsv[:, :, 1].mean())
    return sat < threshold, sat


def cursor_ring(img: np.ndarray) -> float | None:
    """Area of the pale cursor/ink ring near the screen centre, if present."""
    h, w = img.shape[:2]
    roi = img[int(h * 0.30):int(h * 0.72), int(w * 0.34):int(w * 0.66)]
    if roi.size == 0:
        return None
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, np.array([0, 0, 165]), np.array([180, 120, 255]))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    roi_area = float(roi.shape[0] * roi.shape[1])
    for c in cnts:
        a = cv2.contourArea(c)
        if a < 900 or a > 40000 or a > roi_area * 0.25:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        if bh == 0 or not 0.7 < bw / float(bh) < 1.45:
            continue
        if a / float(bw * bh) < 0.55:
            continue
        best = a if best is None else max(best, a)
    return best


def analyse(frames: list[Path], fps: float, scene_delta: float,
            sat_threshold: float = MARKER_SAT_THRESHOLD,
            merge_gap: float = 1.5) -> list[dict]:
    """Walk the sampled frames and record only the moments that MATTER.

    Consecutive scene_change beats closer together than `merge_gap` seconds
    are merged into one: a single pillar growing spans several sampled frames,
    and emitting one beat per frame would bury the route in noise.
    """
    beats: list[dict] = []
    prev = None
    prev_open = False
    for idx, path in enumerate(frames):
        img = cv2.imread(str(path))
        if img is None:
            continue
        t = idx / float(fps)
        is_open, sat = marker_open(img, sat_threshold)
        delta = None if prev is None else float(cv2.absdiff(prev, img).mean())

        kind = None
        if is_open and not prev_open:
            kind = "marker_opened"
        elif prev_open and not is_open:
            kind = "marker_closed"
        elif delta is not None and delta >= scene_delta:
            kind = "scene_change"

        if kind:
            beat = {
                "t": round(t, 2),
                "timestamp": f"{int(t // 60):02d}:{t % 60:05.2f}",
                "kind": kind,
                "frame": str(path),
                "delta": None if delta is None else round(delta, 2),
                "saturation": round(sat, 1),
                "marker_open": is_open,
            }
            if is_open:
                ring = cursor_ring(img)
                beat["cursor_ring_px"] = None if ring is None else int(ring)

            # Merge a run of adjacent scene_changes into the first one, so a
            # pillar growing over 2s is ONE beat, not five.
            last = beats[-1] if beats else None
            if (kind == "scene_change" and last
                    and last["kind"] == "scene_change"
                    and t - last["t"] <= merge_gap):
                last["merged"] = last.get("merged", 1) + 1
                last["delta"] = max(last["delta"] or 0, beat["delta"] or 0)
                last["t_end"] = round(t, 2)
            else:
                beats.append(beat)

        prev, prev_open = img, is_open
    return beats


def infer(beats: list[dict]) -> list[dict]:
    """Attach an INFERRED action to each beat, with its evidence.

    Honesty rule: the video does not contain controller data, so each entry
    says what was observed and what that IMPLIES - it is a hypothesis for the
    agent to try, not a recorded input.
    """
    for i, b in enumerate(beats):
        if b["kind"] == "marker_opened":
            b["inferred_action"] = "hold RT to open the Magic Marker"
            b["evidence"] = (f"frame desaturated to {b['saturation']} "
                             f"(<70 = time-slow marker mode)")
        elif b["kind"] == "marker_closed":
            prev_open = next((x for x in reversed(beats[:i])
                              if x["kind"] == "marker_opened"), None)
            held = (round(b["t"] - prev_open["t"], 2)
                    if prev_open else None)
            b["inferred_action"] = "release RT; a drawing was committed"
            b["evidence"] = (f"palette returned to normal; marker was open "
                             f"for {held}s" if held else "palette normal")
            b["marker_open_seconds"] = held
        else:
            big = b["delta"] or 0
            if big > 40:
                b["inferred_action"] = ("camera panned / new area or a large "
                                        "structure appeared")
            elif big > 12:
                b["inferred_action"] = "Max moved or a drawing grew"
            else:
                b["inferred_action"] = "small on-screen change"
            b["evidence"] = f"frame delta {b['delta']} (ambient is 2-4)"
    return beats


def write_route(beats: list[dict], out: Path, video: Path) -> Path:
    """Emit the route as a prompt fragment the gameplay agent can be given."""
    draws = [b for b in beats if b["kind"] == "marker_closed"]
    lines = [
        "# ROUTE OBSERVED IN A HUMAN PLAYTHROUGH",
        "",
        f"Source: {video.name}",
        "",
        "This is what a human did, in order, on the way to the checkpoint.",
        "Use it as a GUIDE to what the level needs, not as a script: your",
        "own frames are the truth. Timings are when it happened in the",
        "video, so they show ORDER and rough spacing, not button presses.",
        "",
        f"Total beats: {len(beats)}   Magic Marker uses: {len(draws)}",
        "",
        "| # | time | what happened | inferred action |",
        "|---|------|---------------|-----------------|",
    ]
    for i, b in enumerate(beats, 1):
        lines.append(f"| {i} | {b['timestamp']} | {b['kind']} | "
                     f"{b.get('inferred_action', '')} |")
    lines += [
        "",
        "## Magic Marker moments in detail",
        "",
    ]
    for i, b in enumerate(draws, 1):
        held = b.get("marker_open_seconds")
        lines.append(f"{i}. at {b['timestamp']} the marker was open for "
                     f"{held}s -> one drawing was made there.")
    lines += [
        "",
        "## Honesty note",
        "",
        "Button presses are NOT in the video. Every 'inferred action' above",
        "is deduced from pixels: desaturation means the marker opened, a",
        "large delta means the camera panned or a structure appeared. Stick",
        "angles and exact aim are unknowable from video and must still be",
        "solved live from each frame.",
        "",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Turn a playthrough video into a route plan.")
    ap.add_argument("video", type=Path)
    ap.add_argument("--out", type=Path, default=Path("artifacts/route"))
    ap.add_argument("--fps", type=float, default=2.0,
                    help="Frames sampled per second of video (default 2).")
    ap.add_argument("--scene-delta", type=float, default=12.0,
                    help="Delta counted as a real scene change (ambient 2-4).")
    ap.add_argument("--sat-threshold", type=float,
                    default=MARKER_SAT_THRESHOLD,
                    help="Mean saturation below which the marker is treated "
                         "as OPEN. Closed frames measured 70.4-191.0, so the "
                         "default 62 is just under the observed floor.")
    ap.add_argument("--merge-gap", type=float, default=1.5,
                    help="Merge scene changes closer than this many seconds.")
    args = ap.parse_args()

    if not args.video.is_file():
        print(f"No such video: {args.video}")
        return 2

    meta = probe(args.video)
    dur = float(meta.get("duration") or 0)
    print(f"video   : {args.video.name}")
    print(f"size    : {meta.get('width')}x{meta.get('height')}")
    print(f"duration: {dur:.1f}s  -> sampling at {args.fps}/s "
          f"= ~{int(dur * args.fps)} frames")

    frames_dir = args.out / "frames"
    frames = extract(args.video, frames_dir, args.fps)
    print(f"extracted {len(frames)} frames to {frames_dir}")

    beats = infer(analyse(frames, args.fps, args.scene_delta,
                          args.sat_threshold, args.merge_gap))
    print(f"found {len(beats)} key beats "
          f"({sum(1 for b in beats if b['kind'] == 'marker_closed')} "
          f"marker uses)")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "route.json").write_text(
        json.dumps(beats, indent=2), encoding="utf-8")
    route = write_route(beats, args.out / "ROUTE.md", args.video)
    print(f"wrote {route}")
    print(f"wrote {args.out / 'route.json'}")
    for b in beats[:12]:
        print(f"  {b['timestamp']}  {b['kind']:<14} "
              f"{b.get('inferred_action', '')}")
    if len(beats) > 12:
        print(f"  ... and {len(beats) - 12} more")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
