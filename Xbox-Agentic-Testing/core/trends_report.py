"""
trends_report.py - one page across ALL runs: is each scenario getting better?

Each run's report covers that run only. This page reads every
artifacts/runs/*/reports/verification.json (+ execution.json, lessons-applied)
and shows, per scenario: verdict history, criteria-met trend, duration trend,
and which lessons were active. Presentation only - verdicts are read, never
recomputed.

    python -m core.trends_report               # writes artifacts/trends.html
    python console.py trends                   # same, via the CLI
"""

from __future__ import annotations

import html
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from core.professional_report import _CSS


def _load(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def collect(artifacts_dir: Path | str) -> dict[str, list[dict[str, Any]]]:
    """scenario_id -> chronological list of run summaries."""
    runs_dir = Path(artifacts_dir) / "runs"
    by_scenario: dict[str, list[dict[str, Any]]] = {}
    for run in sorted(runs_dir.glob("run-*")) if runs_dir.is_dir() else []:
        reports = run / "reports"
        verification = _load(reports / "verification.json")
        if not verification:
            continue
        execution = _load(reports / "execution.json")
        # Dry runs send no input and judge nothing; counting them as
        # INCONCLUSIVE runs would drag every pass rate down.
        if execution and not execution.get("dispatched_steps"):
            continue
        criteria = verification.get("criteria") or []
        applied = []
        for path in sorted(reports.glob("lessons-applied-r*.json")):
            applied += [str(l.get("id")) for l in _load(path).get("lessons") or []]
        sid = str(verification.get("scenario_id") or execution.get("scenario_id") or "?")
        by_scenario.setdefault(sid, []).append({
            "run_id": run.name,
            "verdict": str(verification.get("verdict", "?")).upper(),
            "met": sum(1 for c in criteria if c.get("met") is True),
            "total": len(criteria),
            "duration": float(execution.get("duration_seconds") or 0),
            "steps": len(execution.get("steps") or []),
            "lessons": sorted(set(applied)),
            "report": (reports / "execution-report.html"),
        })
    return by_scenario


def _spark(values: list[float], top: float, color: str) -> str:
    """Inline SVG line chart; no JS, no external assets."""
    if not values:
        return ""
    w, h = 220, 44
    top = top or max(values) or 1
    step = w / max(1, len(values) - 1)
    pts = " ".join(f"{i * step:.1f},{h - 4 - (v / top) * (h - 8):.1f}"
                   for i, v in enumerate(values))
    dots = "".join(f"<circle cx='{i * step:.1f}' cy='{h - 4 - (v / top) * (h - 8):.1f}' r='2.5' "
                   f"fill='{color}'/>" for i, v in enumerate(values))
    return (f"<svg width='{w}' height='{h}' viewBox='-4 0 {w + 8} {h}'>"
            f"<polyline fill='none' stroke='{color}' stroke-width='2' points='{pts}'/>"
            f"{dots}</svg>")


def render(by_scenario: dict[str, list[dict[str, Any]]], artifacts_dir: Path) -> str:
    e = lambda v: html.escape(str(v))
    cards = []
    for sid, runs in sorted(by_scenario.items(), key=lambda kv: kv[1][-1]["run_id"], reverse=True):
        passes = sum(r["verdict"] == "PASS" for r in runs)
        latest = runs[-1]
        chips = "".join(
            f"<a href='{e(r['report'].relative_to(artifacts_dir).as_posix())}' "
            f"title='{e(r['run_id'])}: {r['met']}/{r['total']} criteria'>"
            f"<span class='tag {e(r['verdict'])}' style='margin:2px'>{e(r['verdict'][:4])}</span></a>"
            if r["report"].is_file() else
            f"<span class='tag {e(r['verdict'])}' style='margin:2px'>{e(r['verdict'][:4])}</span>"
            for r in runs[-20:])
        rows = "".join(
            f"<tr><td><code>{e(r['run_id'])}</code></td>"
            f"<td><span class='tag {e(r['verdict'])}'>{e(r['verdict'])}</span></td>"
            f"<td>{r['met']}/{r['total']}</td><td>{r['steps']}</td>"
            f"<td>{r['duration']:.0f}s</td>"
            f"<td class='small'>{e(', '.join(l[:4] for l in r['lessons'])) or '-'}</td></tr>"
            for r in reversed(runs[-10:]))
        cards.append(f"""
<div class='card' style='margin-bottom:18px'>
  <h3>{e(sid)} <span class='badge {e(latest['verdict'])}'>{e(latest['verdict'])}</span></h3>
  <p class='small'>{len(runs)} run(s) &middot; {passes} pass &middot;
    pass rate {passes / len(runs) * 100:.0f}% &middot; latest {e(latest['run_id'])}</p>
  <div style='margin:8px 0'>{chips}</div>
  <div class='grid two'>
    <div><span class='label'>Criteria met</span><br>
      {_spark([r['met'] for r in runs[-20:]], max(r['total'] for r in runs) or 1, '#16a34a')}</div>
    <div><span class='label'>Duration</span><br>
      {_spark([r['duration'] for r in runs[-20:]], 0, '#7c3aed')}</div>
  </div>
  <table style='margin-top:10px'><thead><tr><th>Run</th><th>Verdict</th><th>Criteria</th>
    <th>Steps</th><th>Duration</th><th>Lessons active</th></tr></thead><tbody>{rows}</tbody></table>
</div>""")
    total_runs = sum(len(v) for v in by_scenario.values())
    total_pass = sum(r["verdict"] == "PASS" for v in by_scenario.values() for r in v)
    return f"""<!doctype html><html><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'>
<title>Execution trends</title><style>{_CSS}</style></head><body><main class='page'>
<section class='hero'><h1>Execution trends</h1>
<p class='subtitle'>Xbox Agentic Testing &middot; all runs &middot; generated
{datetime.now().isoformat(timespec='seconds')}</p>
<div class='grid kpis' style='grid-template-columns:repeat(3,1fr)'>
<div class='card kpi'><span class='label'>Scenarios</span><strong>{len(by_scenario)}</strong></div>
<div class='card kpi g'><span class='label'>Runs</span><strong>{total_runs}</strong></div>
<div class='card kpi p'><span class='label'>Overall pass rate</span>
<strong>{(total_pass / total_runs * 100) if total_runs else 0:.0f}%</strong></div>
</div></section>
<h2>Per scenario</h2>
{''.join(cards) or "<p class='small'>No completed runs found.</p>"}
<footer>Verdicts are read from each run's verification.json and are not
recalculated. Lesson ids show which learned failures each plan was steered by.</footer>
</main></body></html>"""


def build(artifacts_dir: Path | str) -> Path:
    artifacts_dir = Path(artifacts_dir).resolve()
    out = artifacts_dir / "trends.html"
    out.write_text(render(collect(artifacts_dir), artifacts_dir), encoding="utf-8")
    return out


if __name__ == "__main__":
    import sys
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "artifacts"
    print(f"wrote {build(root)}")
