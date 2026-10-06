"""Identify the right stick by UI, not by movement.

Movement-based probing proved unreliable: combat knockback and death/respawn
move the camera regardless of our input. But UI is unambiguous - if clicking
the right stick opens the EMOTE WHEEL, then `rs` is emotes (as
dungeons_controls.yaml says) and cannot be the dodge.

An emote wheel is a large, bright, centred overlay, so it shows up as a big
brightness/structure change in the middle of the screen.
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for sub in ("core", "tools", "agents", "graph"):
    sys.path.insert(0, str(ROOT / sub))

from adapters import HardwareBridge
from config import Config, load_dotenv_if_present
from gameplay_engine import grab_nonblank

load_dotenv_if_present(ROOT / ".env")
settings = Config.load_all(ROOT / "config", {"settings": "settings.yaml"},
                           base=ROOT)["settings"]
hw = HardwareBridge(settings)
cam, pad = hw.capture(), hw.pad()
OUT = ROOT / "scratch" / "rs_probe"
OUT.mkdir(parents=True, exist_ok=True)


def grab(tag, flush=5):
    for _ in range(flush):
        try:
            cam.grab(allow_blank=True)
        except TypeError:
            cam.grab()
    frame, _ = grab_nonblank(cam)
    if frame is not None:
        cv2.imwrite(str(OUT / f"{tag}.jpg"),
                    cv2.resize(frame, (960, 540), interpolation=cv2.INTER_AREA),
                    [cv2.IMWRITE_JPEG_QUALITY, 75])
    return frame


before = grab("before")
print("pressing rs (right stick click)...", flush=True)
ret = pad.press("rs", duration=0.10)
time.sleep(0.6)
after = grab("after")
print("dispatched:", ret, flush=True)

if before is not None and after is not None:
    # Centre region where a radial wheel would appear.
    cb = cv2.cvtColor(before[200:900, 500:1450], cv2.COLOR_BGR2GRAY)
    ca = cv2.cvtColor(after[200:900, 500:1450], cv2.COLOR_BGR2GRAY)
    print(f"centre brightness: before={cb.mean():.1f} after={ca.mean():.1f}")
    print(f"centre abs-diff  : {float(cv2.absdiff(cb, ca).mean()):.2f}")
    # A modal overlay also dims/!blurs the HUD strip at the bottom.
    hb = cv2.cvtColor(before[900:1060, 500:1450], cv2.COLOR_BGR2GRAY)
    ha = cv2.cvtColor(after[900:1060, 500:1450], cv2.COLOR_BGR2GRAY)
    print(f"HUD abs-diff     : {float(cv2.absdiff(hb, ha)).mean() if False else float(cv2.absdiff(hb, ha).mean()):.2f}")
print("inspect scratch/rs_probe/before.jpg vs after.jpg")

# Leave no modal open: press B to back out if a wheel appeared.
pad.press("b", duration=0.08)
