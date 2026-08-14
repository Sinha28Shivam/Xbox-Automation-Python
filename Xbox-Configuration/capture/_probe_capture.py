"""
_probe_capture.py - Prove the AVerMedia capture card works for automation.

We are NOT going to trust "a file was written". A blank/black frame is still a
file. This probe checks the PIXELS and whether the image CHANGES over time -
because a frozen or black feed would silently break every screen check later.

Run:
    python _probe_capture.py
    python _probe_capture.py --backend ffmpeg
    python _probe_capture.py --device "AVerMedia ExtremeCap UVC"
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import time
from pathlib import Path

try:
    import cv2
    import numpy as np
except ImportError:
    sys.exit("Needs opencv + numpy:  pip install opencv-python numpy")

DEVICE = "AVerMedia ExtremeCap UVC"
OUT_DIR = Path(__file__).resolve().parent / "_probe_frames"


def describe(img: "np.ndarray", label: str) -> dict:
    """Report whether a frame actually contains a picture."""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    mean = float(gray.mean())
    std = float(gray.std())
    unique = int(len(np.unique(gray[::8, ::8])))
    # A real picture has spread. Flat colour => std ~0 and few unique values.
    blank = std < 1.0 or unique < 4
    print(f"  {label}: {w}x{h}  mean={mean:6.2f}  std={std:6.2f}  "
          f"tones={unique:3d}  -> {'BLANK/FLAT' if blank else 'has content'}")
    return {"w": w, "h": h, "mean": mean, "std": std,
            "unique": unique, "blank": blank}


def grab_ffmpeg(device: str, out: Path, size: str = "1920x1080") -> bool:
    """Grab one frame with ffmpeg. -vf format=bgr24 avoids MJPEG decode noise."""
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error",
           "-f", "dshow", "-video_size", size, "-framerate", "30",
           "-rtbufsize", "100M",
           "-i", f"video={device}",
           "-frames:v", "1", "-update", "1", "-y", str(out)]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if res.returncode != 0 and not out.exists():
        print("    ffmpeg said:")
        for line in (res.stderr or "").splitlines()[:6]:
            print(f"      {line}")
        return False
    return out.exists()


def probe_opencv(index: int, shots: int = 5) -> bool:
    """Open the device directly with OpenCV and check several frames."""
    print(f"\n--- OpenCV backend (device index {index}) ---")
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("  could not open the device")
        return False

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"  reports {int(w)}x{int(h)} @ {fps:.0f} fps")

    OUT_DIR.mkdir(exist_ok=True)
    frames, stats, times = [], [], []
    for i in range(shots):
        t0 = time.time()
        ok, frame = cap.read()
        dt = (time.time() - t0) * 1000
        if not ok or frame is None:
            print(f"  frame {i}: READ FAILED")
            continue
        times.append(dt)
        frames.append(frame)
        stats.append(describe(frame, f"frame {i} ({dt:5.1f} ms)"))
        cv2.imwrite(str(OUT_DIR / f"opencv_{i}.png"), frame)
        time.sleep(0.4)
    cap.release()

    if not frames:
        print("  RESULT: no frames at all")
        return False

    if times:
        print(f"\n  grab latency: min {min(times):.1f} ms  "
              f"avg {sum(times)/len(times):.1f} ms  max {max(times):.1f} ms")

    all_blank = all(s["blank"] for s in stats)
    print(f"  all frames blank/flat? {all_blank}")

    # Motion check: does the picture change between frames?
    if len(frames) >= 2:
        diffs = []
        for a, b in zip(frames, frames[1:]):
            ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
            gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
            diffs.append(float(cv2.absdiff(ga, gb).mean()))
        print("  frame-to-frame difference: "
              + ", ".join(f"{d:.3f}" for d in diffs))
        if max(diffs) < 0.01:
            print("  NOTE: frames are identical. Either the console shows a")
            print("        static screen (fine) or the feed is frozen (bad).")
        else:
            print("  Frames DIFFER -> the feed is live. This is what the")
            print("  verification layer needs to detect screen changes.")

    print(f"\n  frames saved in: {OUT_DIR}")
    return not all_blank


def probe_ffmpeg(device: str) -> bool:
    print(f"\n--- ffmpeg backend (device '{device}') ---")
    OUT_DIR.mkdir(exist_ok=True)
    ok_any = False
    for i in range(2):
        out = OUT_DIR / f"ffmpeg_{i}.png"
        t0 = time.time()
        got = grab_ffmpeg(device, out)
        dt = time.time() - t0
        if not got:
            print(f"  grab {i}: FAILED")
            continue
        img = cv2.imread(str(out))
        if img is None:
            print(f"  grab {i}: file written but NOT DECODABLE")
            continue
        s = describe(img, f"grab {i} ({dt:.2f}s)")
        ok_any = ok_any or not s["blank"]
        time.sleep(0.5)
    return ok_any


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=DEVICE)
    ap.add_argument("--index", type=int, default=None,
                    help="OpenCV device index (auto-scans if omitted)")
    ap.add_argument("--backend", choices=["opencv", "ffmpeg", "both"],
                    default="both")
    args = ap.parse_args()

    print("=" * 68)
    print("CAPTURE CARD PROBE - checking PIXELS, not just 'a file appeared'")
    print("=" * 68)

    ok = False
    if args.backend in ("opencv", "both"):
        indices = [args.index] if args.index is not None else [0, 1, 2]
        for idx in indices:
            if probe_opencv(idx):
                ok = True
                print(f"\n  >>> OpenCV index {idx} gives real video <<<")
                break
    if args.backend in ("ffmpeg", "both"):
        if probe_ffmpeg(args.device):
            ok = True

    print("\n" + "=" * 68)
    if ok:
        print("RESULT: capture works and produces real image data.")
        print("        Ready to build the perception layer on this.")
    else:
        print("RESULT: no usable image. Check, in this order:")
        print("  1. Is RECentral (or Teams/OBS/Camera) holding the device?")
        print("     Only ONE app can open a capture device at a time.")
        print("  2. Is the console powered on and outputting video?")
        print("  3. Is HDMI going INTO the card's IN port (not OUT)?")
        print("  4. HDCP: Xbox encrypts protected content. Netflix/etc will be")
        print("     black. The dashboard and most games are usually fine.")
    print("=" * 68)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
