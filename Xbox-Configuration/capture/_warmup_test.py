"""Find out how long the card takes to start delivering real frames.

In an earlier run frames 0-1 were blank and content appeared at frame 2, so the
card clearly needs a moment to sync after opening. This measures how long.
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from capture import frame_stats, is_blank  # noqa: E402

INDEX = int(sys.argv[1]) if len(sys.argv) > 1 else 1
MAX_READS = 60

print(f"Opening device index {INDEX} and reading up to {MAX_READS} frames...\n")
cap = cv2.VideoCapture(INDEX, cv2.CAP_DSHOW)
if not cap.isOpened():
    sys.exit("could not open device")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

t_open = time.time()
first_good = None
for i in range(MAX_READS):
    ok, frame = cap.read()
    if not ok or frame is None:
        print(f"  read {i:2d}: FAILED")
        time.sleep(0.1)
        continue
    s = frame_stats(frame)
    blank = is_blank(frame)
    elapsed = time.time() - t_open
    if i < 12 or not blank or i % 10 == 0:
        print(f"  read {i:2d} ({elapsed:5.2f}s): {frame.shape[1]}x{frame.shape[0]} "
              f"std={s['std']:6.2f} tones={int(s['tones']):3d} "
              f"{'BLANK' if blank else '<-- HAS CONTENT'}")
    if not blank and first_good is None:
        first_good = (i, elapsed)
        cv2.imwrite(str(Path(__file__).parent / "_warmup_first_good.png"), frame)
    time.sleep(0.1)

cap.release()

print()
if first_good:
    print(f"RESULT: first real frame at read #{first_good[0]} "
          f"after {first_good[1]:.2f}s")
    print("        -> increase warmup_frames / add a settle delay")
    print("        saved: _warmup_first_good.png")
else:
    print("RESULT: never got content in "
          f"{MAX_READS} reads. Not a warmup problem.")
    print("        The card is delivering a genuinely flat signal.")
