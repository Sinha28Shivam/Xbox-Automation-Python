"""
_verify_report.py - do the report's screenshots actually load?

    python _verify_report.py [run-id]

A Markdown report with broken image links renders perfectly and shows nothing.
Nobody notices until they open it looking for evidence and find empty boxes -
by which point the run is hours old and the console has moved on.

So this resolves every image link the way a Markdown viewer would (relative to
the report file) and reports which ones exist on disk.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "artifacts" / "runs"
IMAGE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")


def main() -> int:
    if len(sys.argv) > 1:
        run_dir = RUNS / sys.argv[1]
    else:
        dirs = sorted(d for d in RUNS.iterdir() if d.is_dir())
        if not dirs:
            print("No runs found.")
            return 1
        run_dir = dirs[-1]

    report = run_dir / "reports" / "report.md"
    if not report.is_file():
        print(f"No report.md in {run_dir}")
        return 1

    text = report.read_text(encoding="utf-8")
    links = IMAGE.findall(text)

    print(f"\n{'=' * 70}\n  REPORT: {report}\n{'=' * 70}\n")
    print(f"  image links     : {len(links)}")
    print(f"  frames on disk  : "
          f"{len(list((run_dir / 'frames').glob('*.png')))}")
    print(f"  has OCR section : {'Text read from this frame' in text}")
    print(f"  has evidence    : {'## Evidence' in text}\n")

    if not links:
        print("  NO IMAGE LINKS. The report has no visual evidence at all.")
        return 1

    broken = 0
    for alt, href in links:
        # Resolve exactly as a Markdown viewer does: relative to the .md file.
        target = (report.parent / href).resolve()
        exists = target.is_file()
        broken += 0 if exists else 1
        print(f"  [{'OK ' if exists else 'DEAD'}] {href}")
        if not exists:
            print(f"         -> {target}")

    print(f"\n{'=' * 70}")
    if broken:
        print(f"  {broken} of {len(links)} images are BROKEN.")
        print("  The report renders, but its evidence is invisible.")
    else:
        print(f"  All {len(links)} images resolve. Evidence is viewable.")
    print("=" * 70 + "\n")
    return 1 if broken else 0


if __name__ == "__main__":
    sys.exit(main())
