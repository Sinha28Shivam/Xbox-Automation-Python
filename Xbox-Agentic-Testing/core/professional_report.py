"""Shared professional HTML dashboard for every execution type.

The dashboard is deliberately presentation-only. Verdicts and status counts
come from the already-produced execution artifacts; this module never judges a
run or changes its status.
"""

from __future__ import annotations

import html
from typing import Any


_CSS = """
:root{--ink:#172033;--muted:#637083;--line:#e4e9f0;--panel:#fff;
--blue:#2563eb;--green:#16a34a;--red:#dc2626;--amber:#d97706;
--slate:#64748b;--purple:#7c3aed}
*{box-sizing:border-box}body{margin:0;background:#f6f8fb;color:var(--ink);
font:14px/1.55 Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
.page{max-width:1240px;margin:auto;padding:34px 28px 64px}
.hero{background:linear-gradient(125deg,#fff 0%,#f4f7ff 58%,#eef7ff 100%);
border:1px solid var(--line);border-radius:18px;padding:28px 30px;
box-shadow:0 8px 28px #1d355711}
h1{font-size:29px;line-height:1.2;margin:0 0 7px;letter-spacing:-.5px}
h2{font-size:18px;margin:30px 0 12px}h3{font-size:15px;margin:0 0 7px}
.subtitle{color:var(--muted);margin:0}.badge{display:inline-block;
padding:4px 12px;border-radius:999px;font-weight:800;font-size:12px;
letter-spacing:.5px;margin-left:10px;vertical-align:4px}
.PASS{background:#dcfce7;color:#166534}.FAIL,.FAILED,.ERROR{background:#fee2e2;color:#991b1b}
.BLOCKED,.INCONCLUSIVE{background:#fef3c7;color:#92400e}
.SKIPPED,.NOTTESTED,.NOT-TESTED,.NOT-ENCOUNTERED{background:#e2e8f0;color:#475569}
.OBSERVED{background:#dbeafe;color:#1d4ed8}.card{background:var(--panel);
border:1px solid var(--line);border-radius:14px;padding:18px;
box-shadow:0 3px 12px #1d355708}.grid{display:grid;gap:14px}
.kpis{grid-template-columns:repeat(6,minmax(0,1fr));margin-top:18px}
.kpi strong{display:block;font-size:24px;margin-top:3px}.label,.muted{color:var(--muted)}
.label{text-transform:uppercase;font-size:11px;font-weight:800;letter-spacing:.7px}
.layout{grid-template-columns:1.25fr .75fr}.two{grid-template-columns:1fr 1fr}
.rate{height:12px;border-radius:99px;background:#e8edf3;overflow:hidden;margin:12px 0 7px}
.rate>i{display:block;height:100%;background:linear-gradient(90deg,#22c55e,#86efac);
border-radius:inherit}.legend{display:flex;gap:16px;flex-wrap:wrap;color:var(--muted);font-size:12px}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:5px}
table{width:100%;border-collapse:collapse}th,td{padding:10px 9px;
border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
th{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.5px}
tr:last-child td{border-bottom:0}.tag{white-space:nowrap;font-size:11px;
font-weight:800;padding:3px 8px;border-radius:999px}.barrow{display:grid;
grid-template-columns:125px 1fr 45px;gap:10px;align-items:center;margin:10px 0}
.bar{height:10px;background:#edf1f5;border-radius:8px;overflow:hidden}.bar i{display:block;
height:100%;border-radius:inherit}.quote{border-left:4px solid var(--blue);
padding:9px 13px;background:#f5f8ff;color:#344054;border-radius:0 8px 8px 0}
code{background:#eef2f7;padding:2px 5px;border-radius:4px;font-size:12px}
.alert{border-left:4px solid var(--red);background:#fff5f5;padding:12px 15px;
border-radius:0 10px 10px 0;margin-top:16px;color:#7f1d1d}
.hero.v-PASS{background:linear-gradient(125deg,#fff 0%,#effdf4 60%,#dcfce7 100%)}
.hero.v-FAIL,.hero.v-FAILED,.hero.v-ERROR{background:linear-gradient(125deg,#fff 0%,#fff5f5 60%,#fee2e2 100%)}
.hero.v-BLOCKED,.hero.v-INCONCLUSIVE{background:linear-gradient(125deg,#fff 0%,#fffbeb 60%,#fef3c7 100%)}
.kpi{border-top:4px solid var(--blue)}.kpi.g{border-top-color:var(--green)}
.kpi.r{border-top-color:var(--red)}.kpi.s{border-top-color:var(--slate)}
.kpi.p{border-top-color:var(--purple)}.kpi.a{border-top-color:var(--amber)}
.donut{width:150px;height:150px;border-radius:50%;margin:8px auto 14px;
display:grid;place-items:center}.donut b{background:#fff;width:104px;height:104px;
border-radius:50%;display:grid;place-items:center;font-size:24px}
.thumb{width:150px;border-radius:6px;border:1px solid var(--line);display:block}
.ok{color:var(--green);font-weight:700}.bad{color:var(--red);font-weight:700}
.small{font-size:12px;color:var(--muted)}.navgrid{display:grid;gap:12px;
grid-template-columns:repeat(auto-fill,minmax(230px,1fr))}
.navcard{border:1px solid var(--line);border-radius:10px;padding:10px;background:#fbfcfe}
.navcard.hit{border-color:#86efac;background:#f0fdf4}
.navcard img{width:100%;border-radius:6px;margin:6px 0}
.ocr{font:11px/1.4 Consolas,monospace;color:#475569;max-height:110px;overflow:auto;
white-space:pre-wrap;background:#fff;border:1px solid var(--line);border-radius:6px;padding:6px}
ul.recs{margin:6px 0 0;padding-left:18px}ul.recs li{margin:5px 0}
footer{color:var(--muted);font-size:12px;border-top:1px solid var(--line);
margin-top:32px;padding-top:15px}@media(max-width:900px){.kpis{grid-template-columns:repeat(3,1fr)}
.layout,.two{grid-template-columns:1fr}}@media(max-width:560px){.page{padding:18px 12px}
.kpis{grid-template-columns:repeat(2,1fr)}}
"""


def _esc(value: Any) -> str:
    return html.escape(str(value if value is not None else ""))


def _status_class(status: str) -> str:
    return status.upper().replace(" ", "-")


def _status_count(counts: dict[str, Any], *names: str) -> int:
    return sum(int(counts.get(name, 0) or 0) for name in names)


def _confidence(value: Any) -> str:
    return "-" if value is None else f"{float(value):.0%}"


def _duration(value: Any) -> str:
    return "-" if value is None else f"{float(value):.2f}s"


def _nav_state(attempt: dict[str, Any]) -> str:
    if attempt.get("matched"):
        return "target reached"
    if attempt.get("text_visible"):
        label = attempt.get("focused_label") or "another item"
        return f"visible, focus on {_esc(label[:40])}"
    if attempt.get("dispatched"):
        return f"pressed {_esc(attempt.get('direction'))}"
    return "observed"


def _delta(value: Any) -> str:
    return "-" if value is None else f"{float(value):.2f}"


def _frame_src(path: Any, run_dir: Any) -> str:
    """Path relative to reports/, so the HTML works when the run folder is copied."""
    if not path:
        return ""
    try:
        from pathlib import Path
        p = Path(str(path))
        if run_dir:
            return "../" + p.resolve().relative_to(Path(str(run_dir)).resolve()).as_posix()
        return p.as_uri()
    except Exception:
        return str(path)


def _img(path: Any, run_dir: Any, cls: str = "thumb") -> str:
    src = _frame_src(path, run_dir)
    if not src:
        return "<span class='small'>no frame</span>"
    return (f"<a href='{_esc(src)}' target='_blank'>"
            f"<img class='{cls}' loading='lazy' src='{_esc(src)}' alt='frame'></a>")


def _scenario_extras(data: dict[str, Any]) -> str:
    """Abort banner, executive summary, RCA, navigation trace and step timeline."""
    run_dir = data.get("run_dir")
    out: list[str] = []

    attempts = data.get("attempts") or []
    if len(attempts) > 1:
        rows = "".join(
            f"<tr><td>Attempt {a.get('revision')}"
            f"{' <span class=small>(final)</span>' if i == len(attempts) - 1 else ''}</td>"
            f"<td><span class='tag {_status_class(a.get('verdict', ''))}'>{_esc(a.get('verdict'))}</span></td>"
            f"<td>{a.get('criteria_met')}/{a.get('criteria_total')}</td>"
            f"<td>{a.get('steps_run')}/{a.get('steps_planned')}</td>"
            f"<td>{float(a.get('duration_seconds') or 0):.1f}s</td>"
            f"<td>{_esc(a.get('abort_reason') or ('completed' if not a.get('aborted') else ''))}</td></tr>"
            for i, a in enumerate(attempts))
        out.append(
            "<h2>Execution attempts</h2><div class='card'><p class='small'>This run "
            "replanned. The headline verdict reflects the final attempt; earlier "
            "attempts are kept so their progress is not lost.</p><table><thead><tr>"
            "<th>Attempt</th><th>Verdict</th><th>Criteria met</th><th>Steps</th>"
            "<th>Duration</th><th>Ended because</th></tr></thead>"
            f"<tbody>{rows}</tbody></table></div>")

    summary = data.get("executive_summary") or {}
    rca = data.get("rca") or {}
    if summary or rca:
        left = "".join(
            f"<div class='kv'><span class='label'>{_esc(label)}</span>"
            f"<div>{_esc(summary.get(key))}</div></div>"
            for key, label in (
                ("what_was_requested", "Requested"),
                ("what_was_attempted", "Attempted"),
                ("strongest_evidence", "Strongest evidence"),
                ("recommended_next_action", "Recommended next action"))
            if summary.get(key))
        recs = "".join(f"<li>{_esc(r)}</li>" for r in rca.get("recommendations") or [])
        right = (
            f"<div class='kv'><span class='label'>Failure class</span><br>"
            f"<span class='tag FAIL'>{_esc(str(rca.get('failure_class', '')).upper())}</span>"
            f" <span class='small'>confidence {_confidence(rca.get('confidence'))}</span></div>"
            f"<div class='kv'><span class='label'>Primary cause</span>"
            f"<div>{_esc(rca.get('primary_cause'))}</div></div>"
            + (f"<span class='label'>Recommendations</span><ul class='recs'>{recs}</ul>"
               if recs else "")) if rca else "<p class='small'>No RCA was required.</p>"
        out.append(
            "<div class='grid two'>"
            f"<section><h2>AI executive summary</h2><div class='card'>"
            f"{left or '<p class=small>No executive summary was saved.</p>'}</div></section>"
            f"<section><h2>Root cause analysis</h2><div class='card'>{right}</div></section>"
            "</div>")

    for trace in data.get("navigation") or []:
        attempts = trace.get("attempts") or []
        found = bool(trace.get("found")) or any(a.get("matched") for a in attempts)
        cards = "".join(
            f"<div class='navcard{' hit' if a.get('matched') else ''}'>"
            f"<b>Attempt {_esc(a.get('attempt'))}</b> "
            f"<span class='small'>{_nav_state(a)}</span>"
            f"{_img(a.get('frame_path'), run_dir, '')}"
            f"<div class='ocr'>{_esc(chr(10).join(a.get('visible_lines') or []))}</div></div>"
            for a in attempts)
        out.append(
            f"<h2>Guided navigation: \"{_esc(trace.get('target_text'))}\" "
            f"<span class='tag {'PASS' if found else 'FAIL'}'>{'FOUND' if found else 'NOT FOUND'}</span></h2>"
            f"<div class='card'><p class='small'>Direction <b>{_esc(trace.get('direction'))}</b>"
            f" &middot; {len(attempts)} screen(s) captured and read &middot; "
            f"<code>{_esc(trace.get('file'))}</code></p><div class='navgrid'>{cards}</div></div>")

    steps = data.get("steps") or []
    if steps:
        rows = "".join(
            f"<tr><td>{_esc(s.get('index'))}</td><td><code>{_esc(s.get('action'))}</code>"
            f"<div class='small'>{_esc(s.get('arguments') or '')}</div></td>"
            f"<td class='{'bad' if s.get('error') else 'ok'}'>{'Error' if s.get('error') else 'OK'}</td>"
            f"<td>{_delta(s.get('screen_delta'))}</td>"
            f"<td>{_esc(s.get('error') or s.get('observation') or '')}</td>"
            f"<td>{_img(s.get('frame_after'), run_dir)}</td></tr>"
            for s in steps)
        out.append(
            "<h2>Step timeline</h2><div class='card'><table><thead><tr><th>#</th>"
            "<th>Action</th><th>Result</th><th>Screen &Delta;</th><th>Observation</th>"
            f"<th>After frame</th></tr></thead><tbody>{rows}</tbody></table></div>")
    return "".join(out)


def render_dashboard(data: dict[str, Any], mode: str = "scenario") -> str:
    """Render a white, responsive dashboard from a normalized report payload."""
    if mode == "dungeons":
        title = data.get("title") or "Minecraft Dungeons execution"
        verdict = str(data.get("verdict") or "UNKNOWN").upper()
        counts = data.get("counts") or {}
        passed = _status_count(counts, "PASS")
        failed = _status_count(counts, "FAIL", "FAILED")
        not_tested = _status_count(counts, "NOT TESTED", "NOT_TESTED",
                                   "NOT ENCOUNTERED")
        attempted = passed + failed + _status_count(counts, "OBSERVED")
        duration = (data.get("run") or {}).get("duration_seconds", 0)
        total = len(data.get("scenarios") or [])
        rows = data.get("scenarios") or []
        ai_summary = data.get("ai_summary") or (
            "The dashboard is generated from the saved Dungeons cycle log. "
            "A PASS requires a later observable result frame.")
        execution_info = [
            ("Run ID", data.get("run_id", "")),
            ("Cycles", (data.get("run") or {}).get("cycles_run", 0)),
            ("Stop reason", (data.get("run") or {}).get("stop_reason", "")),
            ("Mission complete", (data.get("run") or {}).get("mission_complete", False)),
            ("Objective reached", (data.get("run") or {}).get("objective_reached", False)),
            ("Lowest health", f"{(data.get('survival') or {}).get('min_health', 0)}%"),
        ]
        agent_rows = [
            {"title": name, "summary": f"{len(cycles)} cycles", "duration_seconds": None}
            for name, cycles in (data.get("agents") or {}).items()
        ]
        headers = ("Scenario", "Status", "Seen", "Attempted", "Evidence", "Notes")
        row_html = "".join(
            f"<tr><td>{_esc(row.get('scenario'))}</td>"
            f"<td><span class='tag {_status_class(row.get('status',''))}'>{_esc(row.get('status'))}</span></td>"
            f"<td>{len(row.get('seen_cycles') or [])}</td>"
            f"<td>{len(row.get('attempted_cycles') or [])}</td>"
            f"<td>{len(row.get('evidence_cycles') or [])}</td>"
            f"<td>{_esc('; '.join(row.get('notes') or []))}</td></tr>"
            for row in rows)
    elif mode == "mechanics":
        title = f"{data.get('game_name', 'Game')} mechanics execution"
        counts = data.get("counts") or {}
        passed = _status_count(counts, "PASS")
        failed = _status_count(counts, "FAIL", "FAILED")
        not_tested = _status_count(counts, "NOT TESTED", "NOT_TESTED")
        inconclusive = _status_count(counts, "INCONCLUSIVE")
        if failed:
            verdict = "FAIL"
        elif inconclusive:
            verdict = "INCONCLUSIVE"
        elif passed:
            verdict = "PASS"
        else:
            verdict = "NOT TESTED"
        attempted = passed + failed + inconclusive
        duration = (data.get("totals") or {}).get("duration_seconds", 0)
        total = len(data.get("mechanics") or [])
        rows = data.get("mechanics") or []
        ai_summary = (
            "Mechanic outcomes combine deterministic action-log checks with "
            "saved-frame AI diagnostics; INCONCLUSIVE is kept separate from PASS.")
        execution_info = [
            ("Session ID", data.get("session_id", "")),
            ("Level", data.get("level", "")),
            ("Steps", (data.get("totals") or {}).get("steps", 0)),
            ("Deaths", (data.get("totals") or {}).get("deaths", 0)),
            ("Trace available", data.get("has_trace", False)),
            ("Game", data.get("game_name", "")),
        ]
        agent_rows = []
        headers = ("Mechanic", "Status", "Steps", "Evidence")
        row_html = "".join(
            f"<tr><td>{_esc(row.get('title'))}</td>"
            f"<td><span class='tag {_status_class(row.get('verdict',''))}'>{_esc(row.get('verdict'))}</span></td>"
            f"<td>{_esc(', '.join(str(s) for s in row.get('steps') or [])) or '-'}</td>"
            f"<td>{_esc(row.get('ai_note') or row.get('log_note') or '-')}</td></tr>"
            for row in rows)
    else:
        title = data.get("scenario_title") or data.get("scenario_id") or "Scenario execution"
        verdict = str(data.get("verdict") or "UNKNOWN").upper()
        counts = data.get("counts") or {}
        passed = _status_count(counts, "PASS")
        failed = _status_count(counts, "FAILED", "FAIL")
        not_tested = _status_count(counts, "NOT TESTED", "NOT_TESTED")
        attempted = passed + failed
        total = len(data.get("criteria") or [])
        duration = (data.get("totals") or {}).get("duration_seconds", 0)
        rows = data.get("criteria") or []
        ai_summary = data.get("verifier_summary") or (
            "No AI verifier summary was saved for this execution.")
        execution_info = [
            ("Run ID", data.get("run_id", "")),
            ("Scenario ID", data.get("scenario_id", "")),
            ("Console", data.get("console", "")),
            ("Steps", f"{(data.get('totals') or {}).get('steps_run', 0)}/"
                      f"{(data.get('totals') or {}).get('steps_planned', 0)}"),
            ("Dispatched", (data.get("totals") or {}).get("dispatched", 0)),
            ("Replans", (data.get("totals") or {}).get("replans", 0)),
        ]
        agent_rows = data.get("agents") or []
        headers = ("Success criterion", "Status", "Confidence", "Reasoning")
        row_html = "".join(
            f"<tr><td>{_esc(row.get('title'))}</td>"
            f"<td><span class='tag {_status_class(row.get('verdict',''))}'>{_esc(row.get('verdict'))}</span></td>"
            f"<td>{_confidence(row.get('confidence'))}</td>"
            f"<td>{_esc(row.get('reasoning'))}</td></tr>"
            for row in rows)

    observed = passed + failed + _status_count(counts, "OBSERVED")
    rate = (passed / observed * 100) if observed else 0
    total = total or observed + not_tested
    status_bars = [
        ("Passed", passed, "#16a34a"),
        ("Failed", failed, "#dc2626"),
        ("Not tested", not_tested, "#94a3b8"),
    ]
    bars = "".join(
        f"<div class='barrow'><span>{label}</span><div class='bar'><i style='width:"
        f"{(value / max(total, 1)) * 100:.1f}%;background:{color}'></i></div>"
        f"<b>{value}</b></div>" for label, value, color in status_bars)
    info = "".join(f"<div class='kv'><span class='muted'>{_esc(k)}</span><br>"
                   f"<b>{_esc(v)}</b></div>" for k, v in execution_info)
    agents = "".join(
        f"<tr><td>{_esc(a.get('title', a.get('role', 'Unknown agent')))}</td>"
        f"<td>{_esc(a.get('summary', '-'))}</td>"
        f"<td>{_duration(a.get('duration_seconds'))}</td></tr>"
        for a in agent_rows)
    table_head = "".join(f"<th>{_esc(h)}</th>" for h in headers)
    denom = max(total, 1)
    p_end = passed / denom * 100
    f_end = p_end + failed / denom * 100
    donut = (f"conic-gradient(#16a34a 0 {p_end:.1f}%,#dc2626 {p_end:.1f}% {f_end:.1f}%,"
             f"#cbd5e1 {f_end:.1f}% 100%)")
    extras = _scenario_extras(data) if mode == "scenario" else ""
    abort = ""
    if mode == "scenario" and data.get("aborted"):
        abort = (f"<div class='alert'><b>Execution aborted.</b> "
                 f"{_esc(data.get('abort_reason'))}</div>")
    generated = _esc(data.get("generated_at", ""))
    return f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_esc(title)} - Execution Report</title><style>{_CSS}</style></head><body>
<main class="page"><section class="hero v-{_status_class(verdict)}"><h1>{_esc(title)}
<span class="badge {_status_class(verdict)}">{_esc(verdict)}</span></h1>
<p class="subtitle">Xbox Agentic Testing &middot; central execution report &middot; generated {generated}</p>
{abort}
<div class="grid kpis">
<div class="card kpi p"><span class="label">Pass rate</span><strong>{rate:.0f}%</strong></div>
<div class="card kpi g"><span class="label">Passed</span><strong>{passed}</strong></div>
<div class="card kpi r"><span class="label">Failed</span><strong>{failed}</strong></div>
<div class="card kpi s"><span class="label">Not tested</span><strong>{not_tested}</strong></div>
<div class="card kpi a"><span class="label">Duration</span><strong>{float(duration or 0):.1f}s</strong></div>
<div class="card kpi"><span class="label">Total checks</span><strong>{total}</strong></div>
</div></section>
<div class="grid layout"><section><h2>Verifier summary</h2><div class="card quote">{_esc(ai_summary)}</div>
<h2>Execution information</h2><div class="card"><div class="grid two">{info}</div></div></section>
<section><h2>Outcome graph</h2><div class="card"><div class="donut" style="background:{donut}">
<b>{rate:.0f}%</b></div><div class="rate"><i style="width:{rate:.1f}%"></i></div>
<div class="legend"><span><i class="dot" style="background:#16a34a"></i>Pass rate of attempted checks</span>
<span>{observed} attempted</span></div>{bars}</div></section></div>
<h2>Success criteria</h2><div class="card"><table><thead><tr>{table_head}</tr></thead>
<tbody>{row_html or f'<tr><td colspan="{len(headers)}">No result rows were saved.</td></tr>'}</tbody></table></div>
{extras}
<h2>Agent execution</h2><div class="card"><table><thead><tr><th>Agent</th>
<th>Result / activity</th><th>Duration</th></tr></thead><tbody>
{agents or '<tr><td colspan="3">No agent timeline was saved.</td></tr>'}</tbody></table></div>
<footer>Verdicts are read from the execution artifacts and are not recalculated by
the presentation layer. Evidence quality and caveats remain part of the source
report. Report type: {_esc(mode)}.</footer></main></body></html>"""
