"""Per-frame: red fill in candidate heart box + red in bottom corners (damage vignette)."""
import json
from pathlib import Path

import numpy as np
from PIL import Image

root = Path(__file__).resolve().parent.parent / "artifacts/runs/dungeons-fast"
rep = json.loads((root / "reports/dungeons-cycles.json").read_text(encoding="utf-8"))
cyc = {c.get("cycle"): c for c in rep.get("cycles", [])} if isinstance(rep, dict) else {}
for f in sorted((root / "frames").glob("*-before.jpg")):
    a = np.asarray(Image.open(f).convert("RGB")).astype(int)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    red = (r > 120) & (r > g * 1.6) & (r > b * 1.6)
    box = red[900:1040, 620:820]
    # vertical fill profile: fraction of rows (top->bottom) with red
    rows = box.mean(axis=1)
    corners = (red[960:1080, 0:200].mean() + red[960:1080, 1720:1920].mean()) / 2
    n = int(f.name.split("-")[1])
    c = cyc.get(n, {})
    print(f"{f.name} heart={box.mean()*100:5.1f}% rowsfilled={int((rows>0.2).sum()):3d}/140 "
          f"corners={corners*100:5.1f}% scene={c.get('scene','?')}")
