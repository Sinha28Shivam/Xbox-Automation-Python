"""
route_review.py - read a recorded walkthrough with the VISION MODEL and emit
route context the gameplay agent can actually act on.

WHY THIS EXISTS (and why walkthrough_video.py was not enough)
-------------------------------------------------------------
walkthrough_video.py infers beats from pixel statistics. On the first real
recording (artifacts/walkthroughs/sea-of-sand) that produced a route with
"0 Magic Marker uses" - yet frames at t=8.0s and t=19.5s plainly show a
freshly drawn pillar carrying the erasable `X` badge.

MEASURED on that 51s clip:
  * mean saturation never fell below 78.4 (range 78.4 - 155.5).
  * the marker-open threshold was 62, so it could never fire.
  * i.e. opening the marker does NOT globally desaturate this capture path.

A pale-disc detector was tried as a replacement and rejected: it flagged
82,177px of hazy SKY at t=38.0s and pale wooden DEBRIS at t=12.2s, while the
genuine cursor at t=30.0s measured only 10,300px. Sky outscored the real
target, so no area/roundness threshold separates them. This repeats the
project's existing finding (MARKER_FINDINGS.md): bright/pale detectors lock
onto sand and sky, and CV glow detection is unusable here.

So this tool does what the live agent already does successfully - it SHOWS THE
FRAMES TO THE VISION MODEL. The model reports what is in each frame; we
aggregate that into ordered route context.

WHAT IT CANNOT DO
-----------------
Controller input is not in the video. Stick angles, aim positions and aim
times are unrecoverable; they must still be solved live. What the video DOES
give - and what is genuinely hard to discover by trial and error - is the
ORDER of obstacles and WHERE a drawing was required.

USAGE
    python tools/route_review.py artifacts/walkthroughs/sea-of-sand
    python tools/route_review.py <session> --fps 1.0 --batch 6
"""
from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
from pydantic import BaseModel, Field

# `tools/` is added to sys.path by the tool loader at runtime; when run as a
# script we need the repo root importable for `config` / `llm`.
_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_ROOT), str(_ROOT / "core")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ===========================================================================
# What we ask the model for, per sampled frame
# ===========================================================================
class FrameNote(BaseModel):
    """One frame of the walkthrough, as described by the vision model."""

    index: int = Field(description="The frame number given in the label.")
    area: str = Field(
        description="Short name for where Max is, e.g. 'sunlit sand slope', "
                    "'dark canyon', 'vine wall'.")
    marker_cursor_visible: bool = Field(
        description="True ONLY if the round pale Magic Marker CURSOR/reticle "
                    "is on screen (marker is open). Do not count clouds, "
                    "sky, sunlit sand, bones or pale debris.")
    drawing_present: bool = Field(
        description="True if a PLAYER-DRAWN structure (earth pillar, vine, "
                    "branch, water jet) is visible - often tagged with a "
                    "small blue X badge meaning erasable.")
    drawing_kind: str = Field(
        default="",
        description="If drawing_present: pillar | vine | branch | water | "
                    "other. Else empty string.")
    obstacle: str = Field(
        description="The thing blocking forward progress in this frame, or "
                    "'none' if the way is clear.")
    player_visible: bool = Field(
        description="Is Max (small boy, orange/red hair) visible?")
    note: str = Field(
        description="One short sentence: what the player is doing or just "
                    "achieved in this frame.")


class BatchNotes(BaseModel):
    """The model's reading of a batch of frames."""

    frames: list[FrameNote]


# ===========================================================================
# Frame sampling
# ===========================================================================
def sample_frames(video: Path, out_dir: Path, fps: float) -> list[Path]:
    """Extract frames at `fps` with ffmpeg, downscaled for cheap vision calls.

    640px wide keeps a 1fps/51s clip well inside token budgets while leaving
    the cursor (~100px across at 1080p, ~35px here) clearly visible.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("rv_*.jpg"):
        old.unlink()
    subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(video),
         "-vf", f"fps={fps},scale=640:-2", "-q:v", "4",
         str(out_dir / "rv_%05d.jpg")],
        check=True, timeout=1800)
    return sorted(out_dir.glob("rv_*.jpg"))


def encode(path: Path) -> str:
    img = cv2.imread(str(path))
    if img is None:
        return ""
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return base64.b64encode(buf.tobytes()).decode("ascii") if ok else ""


# ===========================================================================
# The vision pass
# ===========================================================================
BATCH_PROMPT = """You are reviewing frames from a HUMAN playthrough of
"Max: The Curse of Brotherhood" (Xbox platformer) to build route notes for an
AI that will play the same level.

You are given {n} frames in chronological order, labelled "frame <index>".
Describe EACH frame; return one entry per frame, in order.

WHAT TO LOOK FOR

Magic Marker CURSOR (marker_cursor_visible):
  A round, pale, hollow reticle/disc that the player steers to aim a drawing.
  It is a UI element and sits ON TOP of the scene, crisp-edged.
  DO NOT mistake these for it - all were false positives in earlier passes:
    - hazy sky or bright clouds near the top of frame
    - sunlit pale sand
    - white bones, planks, or pale wooden debris on the ground
  If you are not sure it is the reticle, answer false.

PLAYER-DRAWN structures (drawing_present):
  Earth pillars, vines, branches or water jets the player created. They look
  newly grown and often carry a small blue "X" badge, which means "erasable"
  - it does NOT mean the player destroyed it.

obstacle: what physically blocks forward progress here - a gap/chasm, a high
  ledge, water, a cliff, an enemy - or "none" if the path is clear.

Be literal and conservative. These notes will be trusted by another agent, so
a confident wrong answer is worse than "none"/false.
"""


def build_reader() -> Any:
    """A structured, vision-capable runnable returning BatchNotes."""
    from config import Config, load_dotenv_if_present
    from llm import LLMFactory, structured

    # The API key normally lives in a .env beside the repo, not the shell.
    load_dotenv_if_present(_ROOT.parent / ".env")
    load_dotenv_if_present(_ROOT / ".env")
    settings = Config.load(_ROOT / "config" / "settings.yaml", base=_ROOT)
    factory = LLMFactory(settings)
    provider = factory.default_provider
    if not factory.supports_vision(provider):
        raise RuntimeError(
            f"LLM provider '{provider}' is not marked supports_vision in "
            f"settings.yaml, so it cannot read walkthrough frames. Use a "
            f"multimodal provider (anthropic / openai / google).")
    return structured(factory.build(provider=provider), BatchNotes)


def read_batch(reader: Any, batch: list[tuple[int, Path]]) -> list[FrameNote]:
    """Show one batch of frames to the model and return its notes."""
    from langchain_core.messages import HumanMessage

    content: list[dict[str, Any]] = [
        {"type": "text", "text": BATCH_PROMPT.format(n=len(batch))}]
    for idx, path in batch:
        b64 = encode(path)
        if not b64:
            continue
        content.append({"type": "text", "text": f"frame {idx}"})
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})

    result = reader.invoke([HumanMessage(content=content)])
    return list(getattr(result, "frames", []))


# ===========================================================================
# Aggregation: per-frame notes -> ordered route segments
# ===========================================================================
def segment(notes: list[FrameNote], fps: float) -> list[dict]:
    """Collapse consecutive frames that describe the same situation.

    A 51s clip at 1fps is 51 notes, which is too granular to be useful as
    prompt context. Frames are merged while the area AND the obstacle stay
    the same, so the output reads as "what the level asked for, in order".
    """
    segs: list[dict] = []
    for n in sorted(notes, key=lambda x: x.index):
        t = n.index / fps
        key = (n.area.strip().lower(), n.obstacle.strip().lower())
        last = segs[-1] if segs else None
        if last and last["_key"] == key:
            last["t_end"] = round(t, 1)
            last["frames"] += 1
            # Any frame in the run seeing the cursor / a drawing counts.
            last["marker_used"] |= n.marker_cursor_visible
            last["drawing_present"] |= n.drawing_present
            if n.drawing_kind and not last["drawing_kind"]:
                last["drawing_kind"] = n.drawing_kind
            if not n.player_visible:
                last["player_lost_frames"] += 1
            continue
        segs.append({
            "_key": key,
            "t_start": round(t, 1),
            "t_end": round(t, 1),
            "frames": 1,
            "area": n.area.strip(),
            "obstacle": n.obstacle.strip(),
            "marker_used": n.marker_cursor_visible,
            "drawing_present": n.drawing_present,
            "drawing_kind": n.drawing_kind,
            "player_lost_frames": 0 if n.player_visible else 1,
            "note": n.note.strip(),
        })
    for s in segs:
        s.pop("_key", None)
        s["duration"] = round(s["t_end"] - s["t_start"] + 1.0 / fps, 1)
    return segs


def ts(seconds: float) -> str:
    return f"{int(seconds // 60):02d}:{seconds % 60:04.1f}"


# ===========================================================================
# Output
# ===========================================================================
def write_route(segs: list[dict], notes: list[FrameNote], out: Path,
                name: str, fps: float) -> Path:
    """Write route context sized to be pasted into the gameplay prompt."""
    draws = [s for s in segs if s["marker_used"]]
    first_draw = [s for s in segs if s["drawing_present"]]

    lines = [
        f"# HUMAN ROUTE: {name}",
        "",
        f"Read by a vision model from {len(notes)} frames sampled at "
        f"{fps}/s. {len(segs)} route segments, "
        f"{len(draws)} with the Magic Marker open.",
        "",
        "## How to use this",
        "",
        "This is the order a human solved the level in. Use it to know WHAT",
        "the level asks for and WHEN a drawing is needed - not as a script.",
        "Your own live frame is always the truth: if it disagrees with the",
        "table below, believe the frame.",
        "",
        "Controller input is NOT in the video. Stick angles, aim positions",
        "and aim times cannot be recovered and must still be solved live.",
        "",
        "## Route",
        "",
        "| # | time | area | obstacle | marker | drawing |",
        "|---|------|------|----------|--------|---------|",
    ]
    for i, s in enumerate(segs, 1):
        lines.append(
            f"| {i} | {ts(s['t_start'])}-{ts(s['t_end'])} | {s['area']} | "
            f"{s['obstacle']} | {'OPEN' if s['marker_used'] else '-'} | "
            f"{s['drawing_kind'] or ('yes' if s['drawing_present'] else '-')} |")

    lines += ["", "## What the level required", ""]
    if draws:
        for i, s in enumerate(draws, 1):
            lines.append(
                f"{i}. At {ts(s['t_start'])} in the {s['area']}, facing "
                f"{s['obstacle']}, the player opened the marker"
                + (f" and a {s['drawing_kind']} was used."
                   if s["drawing_kind"] else "."))
    else:
        lines.append(
            "_The vision pass saw no frame where the marker cursor was "
            "clearly open. Marker use may still have occurred between "
            "sampled frames - raise --fps to check._")

    if first_draw:
        lines += [
            "",
            f"A player-drawn structure is first visible at "
            f"{ts(first_draw[0]['t_start'])}, so the marker is needed early "
            f"in this level - do not spend cycles trying to walk or jump "
            f"past the first obstacle.",
        ]

    lost = [s for s in segs if s["player_lost_frames"]]
    if lost:
        lines += [
            "",
            "## Frames where Max was not visible",
            "",
            "These are moments the reviewer could not see the player (fall, "
            "off-screen, or camera transition). If this happens live, "
            "recover before issuing more movement:",
            "",
        ]
        for s in lost:
            lines.append(f"- {ts(s['t_start'])} in the {s['area']} "
                         f"({s['player_lost_frames']} frame(s))")

    lines += [
        "",
        "## Per-frame notes",
        "",
    ]
    for n in sorted(notes, key=lambda x: x.index):
        flag = "MARKER " if n.marker_cursor_visible else ""
        lines.append(f"- {ts(n.index / fps)} {flag}{n.note}")
    lines.append("")

    out.write_text("\n".join(lines), encoding="utf-8")
    return out


# ===========================================================================
# CLI
# ===========================================================================
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Read a recorded walkthrough with the vision model and "
                    "write route context for the gameplay agent.")
    ap.add_argument("session", type=Path,
                    help="Walkthrough session dir (containing play.mp4) or "
                         "the video file itself.")
    ap.add_argument("--fps", type=float, default=1.0,
                    help="Frames sampled per second of video (default 1.0).")
    ap.add_argument("--batch", type=int, default=6,
                    help="Frames per vision call (default 6).")
    ap.add_argument("--limit", type=int, default=0,
                    help="Only read the first N frames (0 = all).")
    ap.add_argument("--out", type=Path, default=None,
                    help="Where to write ROUTE.md (default: in the session).")
    args = ap.parse_args()

    sess = args.session
    video = sess if sess.is_file() else sess / "play.mp4"
    if not video.is_file():
        print(f"No video found at {video}")
        return 2
    root = video.parent
    name = root.name

    print(f"session : {root}")
    print(f"video   : {video.name}")

    frames = sample_frames(video, root / "review_frames", args.fps)
    if args.limit:
        frames = frames[:args.limit]
    if not frames:
        print("ffmpeg produced no frames.")
        return 1
    print(f"sampled : {len(frames)} frames at {args.fps}/s")

    try:
        reader = build_reader()
    except Exception as exc:
        print(f"cannot build the vision reader: {exc}")
        return 1

    indexed = list(enumerate(frames))
    notes: list[FrameNote] = []
    total = (len(indexed) + args.batch - 1) // args.batch
    for b in range(total):
        batch = indexed[b * args.batch:(b + 1) * args.batch]
        if not batch:
            continue
        print(f"  reading batch {b + 1}/{total} "
              f"(frames {batch[0][0]}-{batch[-1][0]}) ...", flush=True)
        try:
            got = read_batch(reader, batch)
        except Exception as exc:
            # One bad batch must not lose the whole review.
            print(f"    batch failed: {exc}")
            continue
        valid = {i for i, _ in batch}
        notes.extend(n for n in got if n.index in valid)

    if not notes:
        print("The vision model returned no usable notes.")
        return 1
    print(f"read    : {len(notes)}/{len(frames)} frames described")

    segs = segment(notes, args.fps)
    out_dir = args.out or root
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "route_review.json").write_text(
        json.dumps({"name": name, "fps": args.fps,
                    "frames_read": len(notes),
                    "segments": segs,
                    "notes": [n.model_dump() for n in notes]}, indent=2),
        encoding="utf-8")
    route = write_route(segs, notes, out_dir / "ROUTE.md", name, args.fps)

    marker_segs = [s for s in segs if s["marker_used"]]
    print(f"\nwrote {route}")
    print(f"wrote {out_dir / 'route_review.json'}")
    print(f"\n{len(segs)} segments, {len(marker_segs)} with the marker open:")
    for i, s in enumerate(segs, 1):
        mark = " [MARKER]" if s["marker_used"] else ""
        print(f"  {i:2d}. {ts(s['t_start'])}-{ts(s['t_end'])} "
              f"{s['area']} | {s['obstacle']}{mark}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

