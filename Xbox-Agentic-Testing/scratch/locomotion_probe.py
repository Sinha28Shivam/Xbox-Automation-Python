"""Hardware probe: does left-stick movement change the HUD Position? Independent of the model."""
import re
import sys
import time
from pathlib import Path

import pytesseract

ROOT = Path(__file__).resolve().parent.parent
for sub in ("core", "tools", "agents", "graph"):
    sys.path.insert(0, str(ROOT / sub))

from adapters import HardwareBridge
from artifacts import ArtifactStore
from config import Config, load_dotenv_if_present
from registry import ToolContext
from gameplay_engine import grab_nonblank
from PIL import Image
import cv2

load_dotenv_if_present(ROOT / ".env")
settings = Config.load_all(ROOT / "config", {"settings": "settings.yaml"}, base=ROOT)["settings"]
hw = HardwareBridge(settings)
cam = hw.capture()
pad = hw.pad()


def position():
    frame, blank = grab_nonblank(cam)
    if frame is None:
        return None
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(rgb).crop((40, 225, 760, 300)).convert("L")
    txt = pytesseract.image_to_string(im, config="--psm 7").strip()
    return re.findall(r"-?\d+", txt)[:3], txt


tests = [
    (f"move {d} 2s", (lambda d=d: pad.stick("left_stick", direction=d, duration=2.0, strength=1.0)))
    for d in ("up", "right", "down", "left", "up", "up")
]
print("start", position(), flush=True)
for name, fn in tests:
    ret = fn()
    time.sleep(0.6)
    print(name, "->", ret, "pos:", position(), flush=True)

