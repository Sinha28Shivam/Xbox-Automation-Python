"""Capture one frame, flush stale ones, save a small JPEG for eyeballing."""
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
for sub in ("core", "tools", "agents", "graph"):
    sys.path.insert(0, str(ROOT / sub))

from adapters import HardwareBridge
from config import Config, load_dotenv_if_present
from gameplay_engine import grab_nonblank

load_dotenv_if_present(ROOT / ".env")
settings = Config.load_all(ROOT / "config", {"settings": "settings.yaml"},
                           base=ROOT)["settings"]
cam = HardwareBridge(settings).capture()

for _ in range(5):                      # flush the lagging pipeline
    try:
        cam.grab(allow_blank=True)
    except TypeError:
        cam.grab()
frame, blank = grab_nonblank(cam)
out = ROOT / "scratch" / "peek_small.jpg"
if frame is None:
    print("no frame")
else:
    small = cv2.resize(frame, (960, 540), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(out), small, [cv2.IMWRITE_JPEG_QUALITY, 75])
    print(f"saved {out} blank={blank}")
