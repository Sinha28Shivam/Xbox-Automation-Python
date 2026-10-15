"""Find red clusters in the bottom 25% of a frame (grid of 40x20 px cells)."""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

f = Path(sys.argv[1])
a = np.asarray(Image.open(f).convert("RGB")).astype(int)
h, w, _ = a.shape
r, g, b = a[..., 0], a[..., 1], a[..., 2]
red = (r > 120) & (r > g * 1.6) & (r > b * 1.6)
y_start = int(h * 0.75)
cells = []
for y in range(y_start, h, 20):
    for x in range(0, w, 40):
        d = red[y:y + 20, x:x + 40].mean()
        if d > 0.3:
            cells.append((x, y, round(d, 2)))
print(f.name, len(cells), "red cells")
for c in cells[:60]:
    print(c)
