"""Locate the Dungeons heart gauge: red-pixel density in bottom-centre bands per frame."""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

frames = sorted(Path(__file__).resolve().parent.parent.glob("artifacts/runs/dungeons-fast/frames/*-before.jpg"))
for f in frames:
    a = np.asarray(Image.open(f).convert("RGB")).astype(int)
    h, w, _ = a.shape
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    red = (r > 150) & (r > g * 2) & (r > b * 2)
    row = []
    for y0 in np.arange(0.80, 1.0, 0.02):
        band = red[int(h * y0):int(h * (y0 + 0.02)), int(w * 0.44):int(w * 0.56)]
        row.append(f"{band.mean() * 100:4.0f}")
    print(f.name, f"{w}x{h}", " ".join(row))
