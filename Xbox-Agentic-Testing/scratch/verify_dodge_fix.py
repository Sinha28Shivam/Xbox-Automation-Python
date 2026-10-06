"""Verify the fixed `dodge` really rolls the hero, through execute_move().

Drives the profile's own code path (not a hand-rolled pad call) so this tests
what the bot will actually send. Confirms:
  1. the hero is displaced (camera pans), and
  2. no emote wheel / modal opened (centre brightness does not collapse).
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for sub in ("core", "tools", "agents", "graph", "tools/game_profiles"):
    sys.path.insert(0, str(ROOT / sub))

from adapters import HardwareBridge
from artifacts import ArtifactStore
from config import Config, load_dotenv_if_present
from registry import ToolContext
from gameplay_engine import grab_nonblank
from dungeons_profile import DungeonsMove, execute_move

load_dotenv_if_present(ROOT / ".env")
settings = Config.load_all(ROOT / "config", {"settings": "settings.yaml"},
                           base=ROOT)["settings"]
hw = HardwareBridge(settings)
cam = hw.capture()
ctx = ToolContext(hardware=hw,
                  artifacts=ArtifactStore(ROOT / "artifacts", "dodge-verify"),
                  settings=settings)
OUT = ROOT / "scratch" / "dodge_verify"
OUT.mkdir(parents=True, exist_ok=True)

X0, X1, Y0, Y1 = 200, 1500, 120, 820


def grab(tag=None, flush=5):
    for _ in range(flush):
        try:
            cam.grab(allow_blank=True)
        except TypeError:
            cam.grab()
    frame, _ = grab_nonblank(cam)
    if frame is None:
        return None, None, None
    if tag:
        cv2.imwrite(str(OUT / f"{tag}.jpg"),
                    cv2.resize(frame, (960, 540), interpolation=cv2.INTER_AREA),
                    [cv2.IMWRITE_JPEG_QUALITY, 75])
    world = np.float32(cv2.cvtColor(frame[Y0:Y1, X0:X1], cv2.COLOR_BGR2GRAY))
    centre = float(cv2.cvtColor(frame[200:900, 500:1450],
                                cv2.COLOR_BGR2GRAY).mean())
    return frame, world, centre


def pan(a, b):
    win = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
    (dx, dy), _ = cv2.phaseCorrelate(a, b, win)
    return float(np.hypot(dx, dy))


# Clear any modal (emote/menu wheel) left open by an earlier probe, otherwise
# inputs go to the menu instead of the hero and every reading is garbage.
pad = hw.pad()
for _ in range(2):
    pad.press("b", duration=0.08)
    time.sleep(0.4)
_, _, c_start = grab("state-before-start")
print(f"starting centre brightness: {c_start:.1f} "
      f"(a dark value here means a modal is still open)\n")

print("dodge via execute_move() - 4 headings\n")
rows = []
for d in ("up", "right", "down", "left"):
    time.sleep(2.0)                                    # clear the cooldown
    _, w0, c0 = grab(f"{d}-before")
    res = execute_move(ctx, DungeonsMove(action="dodge", direction=d,
                                         purpose="verify"))
    time.sleep(0.5)
    _, w1, c1 = grab(f"{d}-after")
    shift = pan(w0, w1) if w0 is not None and w1 is not None else -1.0
    drop = (c0 - c1) / max(c0, 1e-6)
    rows.append((d, res.get("dispatched"), shift, c0, c1, drop))
    print(f"  {d:<6} dispatched={res.get('dispatched')} "
          f"pan={shift:6.1f}px centre {c0:5.1f}->{c1:5.1f} "
          f"(drop {drop*100:+.0f}%)", flush=True)

print("\n=== VERDICT ===")
moved = [r for r in rows if r[2] > 20]
modal = [r for r in rows if r[5] > 0.30]
print(f"headings that displaced the hero: {len(moved)}/4")
print(f"headings that opened a modal    : {len(modal)}/4 "
      f"(expect 0 - a wheel would dim the centre ~>30%)")
if len(moved) >= 3 and not modal:
    print("PASS: LB+left_stick rolls the hero, no emote wheel.")
elif modal:
    print("FAIL: something opened a modal overlay.")
else:
    print("INCONCLUSIVE: little movement - may be walled in, in combat, or "
          "mid-cooldown. Inspect scratch/dodge_verify/*.jpg")
