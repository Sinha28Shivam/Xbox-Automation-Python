import glob
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.game_profiles.dungeons_profile import _pixel_health  # noqa: E402

for p in sorted(glob.glob("artifacts/runs/dungeons-fast/frames/dungeons-*-before.jpg")):
    print(p.rsplit("-", 2)[-2], _pixel_health(cv2.imread(p)))
