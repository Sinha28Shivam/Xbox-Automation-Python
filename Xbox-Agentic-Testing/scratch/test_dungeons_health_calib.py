"""Calibrate _pixel_health against saved run frames.

Frame 017 exposed the failure this guards: the model reported "health
critically low (30%)" while the heart gauge was full, and _annotate silently
replaced the claim with no record. Here we read every saved frame and compare
the measured gauge against what the model wrote in the cycle report, so the
pixel reader can be trusted before it overrides the model on hardware.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for sub in ("core", "tools", "agents", "graph", "tools/game_profiles"):
    sys.path.insert(0, str(ROOT / sub))

try:
    import cv2
except ImportError:
    print("SKIP: opencv not installed")
    sys.exit(0)

import game_profiles.dungeons_profile as dp

RUNS = ROOT / "artifacts" / "runs"


def model_health_by_cycle(run: Path) -> dict[int, int]:
    """The model's OWN health claim, recovered from its reasoning text.

    The stored health_percent is useless for this check: _annotate had already
    overwritten it with the pixel value before the report was written, which is
    precisely how frame 017's hallucination stayed hidden. The reasoning string
    is the only surviving record of what the model actually believed.
    """
    report = run / "reports" / "dungeons-cycles.json"
    if not report.exists():
        return {}
    data = json.loads(report.read_text(encoding="utf-8"))
    cycles = data.get("cycles", data) if isinstance(data, dict) else data
    out = {}
    for entry in cycles:
        if not (isinstance(entry, dict) and "cycle" in entry):
            continue
        m = re.search(r"(\d{1,3})\s*%", str(entry.get("reasoning", "")))
        if m:
            pct = int(m.group(1))
            if 0 <= pct <= 100:
                out[entry["cycle"]] = pct
    return out


total = checked = disagreements = unreadable = 0
for run in sorted(RUNS.glob("dungeons*")):
    frames = sorted((run / "frames").glob("*-before.jpg"))
    if not frames:
        continue
    claimed = model_health_by_cycle(run)
    print(f"\n{run.name}: {len(frames)} frames")
    for f in frames:
        total += 1
        frame = cv2.imread(str(f))
        hp = dp._pixel_health(frame)
        if hp is None:
            unreadable += 1
            print(f"  {f.name}: UNREADABLE")
            continue
        # dungeons-017-before.jpg -> 17
        try:
            cycle = int(f.name.split("-")[1])
        except (IndexError, ValueError):
            cycle = None
        said = claimed.get(cycle)
        checked += 1
        if said is not None and abs(said - hp) >= dp._HEALTH_DISAGREE:
            disagreements += 1
            print(f"  {f.name}: pixel={hp:3d}%  model={said:3d}%  <-- DISAGREE")

print(f"\nframes={total} readable={checked} unreadable={unreadable} "
      f"disagreements={disagreements}")
assert total > 0, "no frames found"
# Every frame must yield a usable reading; a None here means the HUD box is
# mis-calibrated and the monitor would silently stop drinking potions.
assert unreadable == 0, f"{unreadable} frames unreadable"
print("pixel health   : OK")
