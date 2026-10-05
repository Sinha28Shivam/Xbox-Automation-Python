import glob
import sys

import cv2

for sub in ("core", "tools", "agents", "graph", "tools/game_profiles"):
    sys.path.insert(0, sub)
from game_profiles.dungeons_profile import _pixel_health  # noqa: E402

for p in sorted(glob.glob("artifacts/runs/dungeons-fast/frames/dungeons-*-before.jpg")):
    print(p.rsplit("-", 2)[-2], _pixel_health(cv2.imread(p)))
