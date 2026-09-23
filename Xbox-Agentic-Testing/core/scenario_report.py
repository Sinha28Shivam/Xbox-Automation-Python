"""
scenario_report.py - a human-readable report for ANY `console.py run --file`
scenario run (Minecraft, Max, or anything else), as opposed to
gameplay/mechanics_report.py which is hardcoded to Max's gameplay-session
format (trace.json / session.json under artifacts/gameplay/).

A scenario run instead writes scenario.json, execution.json, verification.json
and logs/state-digest.json under artifacts/runs/<run-id>/reports and /logs.
This module reads those and produces the same PASS/FAILED/NOT-TESTED-style
report, but keyed on the scenario's own success_criteria instead of a fixed
mechanic list, and with an explicit per-agent breakdown (which agent ran,
what it reported, how long it took) pulled from state-digest.json.

USAGE
    python -m core.scenario_report artifacts/runs/run-20260923-220303
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from html import escape as e
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_ROOT), str(_ROOT / "core"), str(_ROOT / "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Agents run in this fixed pipeline order; state-digest.json's message list
# is chronological but interleaves "started"/"completed" notes, so the
# canonical order is asserted here rather than inferred from the log.
AGENT_ORDER = ("health", "scenario_validator", "planner", "executor",
              "verifier", "rca", "reporter")

AGENT_TITLES = {
    "health": "Health Agent",
    "scenario_validator": "Scenario Validator Agent",
    "planner": "Planner Agent",
    "executor": "Executor Agent",
    "verifier": "Verifier Agent",
    "rca": "RCA Agent",
    "reporter": "Reporter Agent",
}


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_run(run_dir: Path) -> dict[str, Any]:
    """Read every report/log file a scenario run writes. Any may be absent."""
    run_dir = Path(run_dir)
    reports = run_dir / "reports"
    return {
        "run_dir": run_dir,
        "scenario": _load_json(reports / "scenario.json"),
        "plan": _load_json(reports / "plan-r1.json"),
        "execution": _load_json(reports / "execution.json"),
        "verification": _load_json(reports / "verification.json"),
        "health": _load_json(reports / "health.json"),
        "rca": _load_json(reports / "rca.json"),
        "state_digest": _load_json(run_dir / "logs" / "state-digest.json"),
    }


def _agent_timeline(state_digest: dict[str, Any]) -> list[dict[str, Any]]:
    """One entry per agent that ran: role, summary message, duration."""
    messages = state_digest.get("messages") or []
    outputs = state_digest.get("agent_outputs") or {}

    by_role: dict[str, dict[str, Any]] = {}
    for m in messages:
        role = str(m.get("role", ""))
        text = str(m.get("text", ""))
        if not role:
            continue
        entry = by_role.setdefault(role, {"role": role, "summary": "",
                                          "duration_seconds": None})
        if text.startswith("completed in"):
            try:
                entry["duration_seconds"] = float(
                    text.replace("completed in", "").replace("s", "").strip())
            except ValueError:
                pass
        elif text and not entry["summary"]:
            entry["summary"] = text

    timeline: list[dict[str, Any]] = []
    seen: set[str] = set()
    for role in AGENT_ORDER:
        if role in by_role:
            entry = by_role[role]
            entry["title"] = AGENT_TITLES.get(role, role.replace("_", " ").title())
            entry["output"] = outputs.get(role, {})
            timeline.append(entry)
            seen.add(role)
    # Any agent not in the canonical order (custom pipelines) still shown.
    for role, entry in by_role.items():
        if role not in seen:
            entry["title"] = AGENT_TITLES.get(role, role.replace("_", " ").title())
            entry["output"] = outputs.get(role, {})
            timeline.append(entry)
    return timeline


def _criterion_verdict(criterion: dict[str, Any]) -> str:
    if criterion.get("met") is True:
        return "PASS"
    if criterion.get("met") is False:
        return "FAILED"
    return "NOT TESTED"


def analyse(run_dir: Path | str) -> dict[str, Any]:
    """Build the full report payload from a scenario run's artifacts."""
    run_dir = Path(run_dir)
    data = load_run(run_dir)
    scenario = data["scenario"]
    execution = data["execution"]
    verification = data["verification"]
    plan = data["plan"]

    criteria = verification.get("criteria") or []
    results = [{
        "title": c.get("criterion", ""),
        "verdict": _criterion_verdict(c),
        "reasoning": c.get("reasoning", ""),
        "confidence": c.get("confidence"),
        "evidence": c.get("evidence") or [],
    } for c in criteria]

    counts: dict[str, int] = {}
    for r in results:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1

    steps = execution.get("steps") or []
    agents = _agent_timeline(data["state_digest"])

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "scenario_id": scenario.get("id", execution.get("scenario_id", "")),
        "scenario_title": scenario.get("title", ""),
        "scenario_goal": scenario.get("goal", ""),
        "console": scenario.get("console", ""),
        "verdict": str(verification.get("verdict", "unknown")).upper(),
        "verifier_summary": verification.get("summary", ""),
        "not_proven": verification.get("not_proven") or [],
        "totals": {
            "steps_run": len(steps),
            "steps_planned": execution.get("total_steps", len(steps)),
            "dispatched": execution.get("dispatched_steps", 0),
            "duration_seconds": execution.get("duration_seconds", 0.0),
            "replans": plan.get("revision", 1) - 1 if plan else 0,
            "cached_route": bool((data["state_digest"].get("agent_outputs", {})
                                  .get("planner", {}) or {}).get("cached_route")),
        },
        "counts": counts,
        "criteria": results,
        "steps": steps,
        "agents": agents,
    }


# ===========================================================================
# Markdown
# ===========================================================================
def markdown(r: dict[str, Any]) -> str:
    out: list[str] = []
    a = out.append

    a(f"# Scenario Report: {r['scenario_title'] or r['scenario_id']}")
    a("")
    a(f"Run `{r['run_id']}` &middot; verdict **{r['verdict']}** &middot; "
      f"{r['totals']['steps_run']}/{r['totals']['steps_planned']} steps "
      f"&middot; {float(r['totals']['duration_seconds']):.0f}s")
    if r["totals"]["cached_route"]:
        a("Planner replayed a pinned/cached route - 0ms planning latency, "
          "no LLM call.")
    a("")
    if r["scenario_goal"]:
        a(f"**Goal:** {r['scenario_goal']}")
        a("")
    if r["verifier_summary"]:
        a(f"**Verifier summary:** {r['verifier_summary']}")
        a("")

    a("## Success criteria")
    a("")
    for i, c in enumerate(r["criteria"], 1):
        a(f"{i}. {c['title']} - **{c['verdict']}**")
    a("")
    counts = r["counts"]
    a(f"{counts.get('PASS', 0)} passed, {counts.get('FAILED', 0)} failed, "
      f"{counts.get('NOT TESTED', 0)} not tested.")
    a("")

    a("## Agents that ran this scenario")
    a("")
    a("| Agent | Result | Duration |")
    a("|---|---|---|")
    for agent in r["agents"]:
        dur = (f"{agent['duration_seconds']:.2f}s"
               if agent["duration_seconds"] is not None else "-")
        a(f"| {agent['title']} | {agent['summary'] or '-'} | {dur} |")
    a("")

    a("## Criterion detail")
    a("")
    for i, c in enumerate(r["criteria"], 1):
        a(f"### {i}. {c['title']} - {c['verdict']}")
        a("")
        if c["reasoning"]:
            a(f"> {c['reasoning']}")
            a("")
        if c["confidence"] is not None:
            a(f"Confidence: {c['confidence']:.2f}")
            a("")

    a("---")
    a(f"Generated {r['generated_at']} from `{r['run_dir']}`.")
    return "\n".join(out)


# ===========================================================================
# HTML
# ===========================================================================
_CSS = """
body{background:#0d0f13;color:#e6e6e6;font:15px/1.55 -apple-system,
 Segoe UI,Roboto,sans-serif;margin:0}
.wrap{max-width:920px;margin:0 auto;padding:34px 24px 60px}
h1{font-size:24px;margin:0 0 6px}
h2{font-size:19px;margin:34px 0 12px;border-bottom:1px solid #262b36;
 padding-bottom:6px}
h3{font-size:16px;margin:26px 0 10px}
.sub{color:#9aa4b2;margin:0 0 22px}
ol.summary{padding-left:24px;margin:0 0 20px}
ol.summary li{margin:7px 0;font-size:16px}
.tag{display:inline-block;padding:2px 9px;border-radius:11px;font-size:12px;
 font-weight:700;letter-spacing:.4px;margin-left:8px;vertical-align:1px}
.PASS{background:#123d20;color:#5ddc7f;border:1px solid #1e5c31}
.FAILED{background:#42151a;color:#ff8b8b;border:1px solid #6b2028}
.NOTTESTED{background:#2b2f3a;color:#a9b3c4;border:1px solid #3b4252}
.card{background:#161a22;border:1px solid #262b36;border-radius:9px;
 padding:16px 18px;margin:14px 0}
.kv{color:#9aa4b2;font-size:14px;margin:5px 0}
.kv b{color:#e6e6e6;font-weight:600}
.quote{border-left:3px solid #3b4252;padding:7px 13px;margin:11px 0;
 color:#b9c1cd;background:#12151c;font-size:14px}
table{border-collapse:collapse;width:100%;margin:10px 0;font-size:14px}
th,td{border:1px solid #262b36;padding:7px 10px;text-align:left}
th{background:#171b24;color:#9aa4b2}
footer{margin-top:44px;color:#7f8896;font-size:13px;border-top:1px solid
 #262b36;padding-top:16px}
"""


def html(r: dict[str, Any]) -> str:
    t, c = r["totals"], r["counts"]
    o: list[str] = []
    a = o.append

    a(f"<!doctype html><html><head><meta charset='utf-8'>"
      f"<title>Scenario Report - {e(r['scenario_title'] or r['scenario_id'])}"
      f"</title><style>{_CSS}</style></head><body><div class='wrap'>")
    a(f"<h1>{e(r['scenario_title'] or r['scenario_id'])}"
      f"<span class='tag {e(r['verdict'])}'>{e(r['verdict'])}</span></h1>")
    a(f"<p class='sub'>Run <code>{e(r['run_id'])}</code> &middot; "
      f"{t['steps_run']}/{t['steps_planned']} steps &middot; "
      f"{float(t['duration_seconds']):.0f}s"
      + (" &middot; cached route replayed (0ms planning, no LLM call)"
         if t["cached_route"] else "") + "</p>")
    if r["scenario_goal"]:
        a(f"<p class='kv'><b>Goal:</b> {e(r['scenario_goal'])}</p>")
    if r["verifier_summary"]:
        a(f"<p class='kv'><b>Verifier summary:</b> "
          f"{e(r['verifier_summary'])}</p>")

    a("<h2>Success criteria</h2><ol class='summary'>")
    for cr in r["criteria"]:
        cls = cr["verdict"].replace(" ", "")
        a(f"<li>{e(cr['title'])}<span class='tag {cls}'>{e(cr['verdict'])}"
          f"</span></li>")
    a("</ol>")
    a(f"<p class='sub'><b>{c.get('PASS', 0)} passed, {c.get('FAILED', 0)} "
      f"failed, {c.get('NOT TESTED', 0)} not tested.</b></p>")

    a("<h2>Agents that ran this scenario</h2>")
    a("<table><tr><th>Agent</th><th>Result</th><th>Duration</th></tr>")
    for agent in r["agents"]:
        dur = (f"{agent['duration_seconds']:.2f}s"
               if agent["duration_seconds"] is not None else "-")
        a(f"<tr><td>{e(agent['title'])}</td><td>{e(agent['summary'] or '-')}"
          f"</td><td>{dur}</td></tr>")
    a("</table>")

    a("<h2>Criterion detail</h2>")
    for i, cr in enumerate(r["criteria"], 1):
        cls = cr["verdict"].replace(" ", "")
        a(f"<div class='card'><h3>{i}. {e(cr['title'])}"
          f"<span class='tag {cls}'>{e(cr['verdict'])}</span></h3>")
        if cr["reasoning"]:
            a(f"<div class='quote'>{e(cr['reasoning'])}</div>")
        if cr["confidence"] is not None:
            a(f"<div class='kv'><b>Confidence:</b> {cr['confidence']:.2f}"
              f"</div>")
        a("</div>")

    a(f"<footer>Generated {e(r['generated_at'])} from "
      f"<code>{e(r['run_dir'])}</code>.<br>Every verdict comes from the "
      f"verifier agent's own criteria evaluation; this report only "
      f"re-presents it alongside which agent produced each result.</footer>")
    a("</div></body></html>")
    return "".join(o)


# ===========================================================================
# Writing
# ===========================================================================
def build_report(run_dir: Path | str) -> dict[str, str]:
    """Analyse a scenario run and write scenario-report.{md,html,json}."""
    run_dir = Path(run_dir)
    payload = analyse(run_dir)

    out_dir = run_dir / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}

    for fmt, name, render in (
        ("markdown", "scenario-report.md", lambda: markdown(payload)),
        ("html", "scenario-report.html", lambda: html(payload)),
        ("json", "scenario-report.json",
         lambda: json.dumps(payload, indent=2, default=str)),
    ):
        try:
            path = out_dir / name
            path.write_text(render(), encoding="utf-8")
            written[fmt] = str(path)
        except Exception as exc:
            print(f"  [report] {fmt} failed: {exc}", flush=True)

    return written


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        description="Build a scenario report (any console.py run --file run).")
    ap.add_argument("run_dir", type=Path,
                    help="a run dir under artifacts/runs/")
    args = ap.parse_args(argv)

    if not args.run_dir.is_dir():
        print(f"ERROR: not a directory: {args.run_dir}")
        return 2

    written = build_report(args.run_dir)
    if not written:
        print("No report was written.")
        return 1
    for fmt, path in written.items():
        print(f"wrote {fmt:9s} {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
