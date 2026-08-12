"""Verify the docs set: list files and check every internal .md link resolves."""
import re
from pathlib import Path

d = Path(__file__).resolve().parent

print("=== FILES ===")
mds = sorted(d.glob("*.md"))
total = 0
for f in mds:
    lines = len(f.read_text(encoding="utf-8").splitlines())
    total += lines
    print(f"  {f.name:<42} {f.stat().st_size:>6} bytes  {lines:>4} lines")
print(f"  {'TOTAL':<42} {'':>6}         {total:>4} lines")

print("\n=== LINK CHECK ===")
bad = 0
for f in mds:
    text = f.read_text(encoding="utf-8")
    for target in re.findall(r"\]\((?!http)([^)]+\.md)\)", text):
        if not (d / target).exists():
            print(f"  BROKEN: {f.name} -> {target}")
            bad += 1
print("  All internal links resolve OK" if bad == 0 else f"  {bad} broken link(s)")
