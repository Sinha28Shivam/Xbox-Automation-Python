"""
_inspect_run.py - summarise the most recent run's artifacts.

    python _inspect_run.py            # latest run
    python _inspect_run.py <run-id>   # a specific one

Useful when a run's console output has scrolled away, and as a quick way to see
what each agent actually produced without opening five JSON files by hand.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

RUNS = Path(__file__).resolve().parent / "artifacts" / "runs"


def load(path: Path):
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def main() -> int:
    if not RUNS.is_dir():
        print("No runs yet.")
        return 1

    if len(sys.argv) > 1:
        run_dir = RUNS / sys.argv[1]
    else:
        dirs = sorted(d for d in RUNS.iterdir() if d.is_dir())
        if not dirs:
            print("No runs yet.")
            return 1
        run_dir = dirs[-1]

    print(f"\n{'=' * 70}\n  RUN: {run_dir.name}\n{'=' * 70}")

    reports = run_dir / "reports"

    health = load(reports / "health.json")
    if health:
        print(f"\nHEALTH  healthy={health['healthy']}  "
              f"gimx={health['gimx_reachable']}  "
              f"signal={health['capture_has_signal']}")

    scenario = load(reports / "scenario.json")
    if scenario:
        print(f"\nSCENARIO  valid={scenario['valid']}  id={scenario['id']}")
        print(f"  title: {scenario['title']}")
        print(f"  goal:  {scenario.get('goal', '')[:100]}")
        for c in scenario.get("success_criteria", []):
            print(f"    [{c['check_type']}] {c['description'][:80]}")
        for issue in scenario.get("issues", []):
            print(f"    ISSUE: {issue}")

    for plan_path in sorted(reports.glob("plan-r*.json")):
        plan = load(plan_path)
        if plan:
            print(f"\nPLAN r{plan['revision']}  {len(plan['steps'])} steps")
            for s in plan["steps"]:
                print(f"  {s['index']:2d}. {s['action']}({_args(s['arguments'])})")
                print(f"      expect: {s['expected_observation'][:80]}")

    execution = load(reports / "execution.json")
    if execution:
        print(f"\nEXECUTION  completed={execution['completed']}  "
              f"aborted={execution['aborted']}")
        for s in execution["steps"]:
            delta = "n/a" if s["screen_delta"] is None else f"{s['screen_delta']:.3f}"
            print(f"  {s['index']:2d}. {s['action']:22s} "
                  f"dispatched={str(s['dispatched']):5s} delta={delta}")
        if execution.get("abort_reason"):
            print(f"  ABORT: {execution['abort_reason']}")

    verification = load(reports / "verification.json")
    if verification:
        print(f"\nVERIFICATION  {verification['verdict'].upper()}")
        print(f"  {verification['summary'][:200]}")
        for item in verification.get("not_proven", []):
            print(f"  not proven: {item[:100]}")

    rca = load(reports / "rca.json")
    if rca:
        print(f"\nRCA  {rca['failure_class']}  confidence={rca['confidence']}")
        print(f"  {rca['primary_cause'][:200]}")

    report = load(reports / "report.json")
    if report:
        print(f"\n{'=' * 70}\n  VERDICT: {report['verdict'].upper()}\n{'=' * 70}")

    errors = run_dir / "logs" / "agent-errors.log"
    if errors.is_file():
        text = errors.read_text(encoding="utf-8", errors="replace")
        headers = [l for l in text.splitlines() if l.startswith("[")]
        if headers:
            print("\nAGENT ERRORS:")
            for line in headers:
                print(f"  {line[:150]}")

    print()
    return 0


def _args(arguments: dict) -> str:
    return ", ".join(f"{k}={v!r}" for k, v in list(arguments.items())[:3])


if __name__ == "__main__":
    sys.exit(main())
