"""
report_tools.py - write the run's output files.

Three formats, each for a different reader:

  json      machine-readable, the complete record - for dashboards and trends
  markdown  what a human opens, with screenshots linked inline
  junit     what CI consumes, so a red build means a real console failure

WHY BLOCKED IS NOT A FAILURE IN JUNIT
-------------------------------------
JUnit has <failure> and <skipped>. A BLOCKED run - dead capture card, GIMX not
running - is emitted as <skipped>, not <failure>. Reporting a broken rig as a
test failure trains everyone to ignore red builds, and once that happens the
suite has no value. A skipped test with a loud reason keeps the signal clean.
"""

from __future__ import annotations

import json
from typing import Any
from html import escape as html_escape
from xml.sax.saxutils import escape, quoteattr

from registry import ToolContext, ToolSpec, fail, make_tool, ok


def _write_json_report(ctx: ToolContext) -> Any:
    def run(report: dict[str, Any], filename: str = "report.json") -> dict[str, Any]:
        path = ctx.artifacts.save_json(filename, report)
        if path is None:
            return fail("Artifact saving is disabled")
        return ok(path=path, format="json")

    return make_tool(
        run, "write_json_report",
        "Write the complete run record as JSON - the machine-readable "
        "artifact for dashboards and trend analysis.")


def _write_markdown_report(ctx: ToolContext) -> Any:
    def run(report: dict[str, Any],
            filename: str = "report.md") -> dict[str, Any]:
        path = ctx.artifacts.save_text(
            filename, _markdown(ctx, report), subdir="reports")
        if path is None:
            return fail("Artifact saving is disabled")
        return ok(path=path, format="markdown")

    return make_tool(
        run, "write_markdown_report",
        "Write the human-facing Markdown report with screenshots linked "
        "inline and an explicit list of what was not proven.")


def _write_html_report(ctx: ToolContext) -> Any:
    def run(report: dict[str, Any],
            filename: str = "report.html") -> dict[str, Any]:
        path = ctx.artifacts.save_text(
            filename, _html(ctx, report), subdir="reports")
        if path is None:
            return fail("Artifact saving is disabled")
        return ok(path=path, format="html")

    return make_tool(
        run, "write_html_report",
        "Write a styled HTML executive report with screenshots, summary, "
        "requirement context, RCA, and step evidence.")


def _write_junit_report(ctx: ToolContext) -> Any:
    def run(report: dict[str, Any],
            filename: str = "junit.xml") -> dict[str, Any]:
        path = ctx.artifacts.save_text(
            filename, _junit(report), subdir="reports")
        if path is None:
            return fail("Artifact saving is disabled")
        return ok(path=path, format="junit")

    return make_tool(
        run, "write_junit_report",
        "Write JUnit XML for CI. Blocked runs are emitted as 'skipped', not "
        "'failure', so infrastructure problems never masquerade as product "
        "defects.")


def _list_artifacts(ctx: ToolContext) -> Any:
    def run() -> dict[str, Any]:
        return ok(run_dir=str(ctx.artifacts.run_dir),
                  files=ctx.artifacts.files,
                  frames=ctx.artifacts.list_frames())

    return make_tool(
        run, "list_artifacts",
        "List every file produced during this run: frames, logs and reports.")


# ===========================================================================
# Markdown
# ===========================================================================
def _markdown(ctx: ToolContext, r: dict[str, Any]) -> str:
    verdict = str(r.get("verdict", "unknown")).upper()
    icon = {"PASS": "PASS", "FAIL": "FAIL", "BLOCKED": "BLOCKED",
            "INCONCLUSIVE": "INCONCLUSIVE"}.get(verdict, verdict)

    out: list[str] = [
        f"# Test Report - {r.get('scenario_title') or r.get('scenario_id', '')}",
        "",
        f"**Verdict:** {icon}",
        f"**Run:** `{r.get('run_id', '')}`",
        f"**Duration:** {r.get('duration_seconds', 0):.1f}s",
        f"**Finished:** {r.get('finished_at', '')}",
        "",
        "## Summary",
        "",
        str(r.get("summary", "")).strip() or "_No summary provided._",
        "",
    ]
    if r.get("requirement_id"):
        out.insert(4, f"**Requirement:** `{r.get('requirement_id')}` - "
                      f"{r.get('requirement_title', '')}")

    executive = r.get("executive_summary") or {}
    if executive:
        out += ["## Executive Summary", ""]
        if r.get("requirement_goal"):
            out += [f"**Requested:** {str(executive.get('what_was_requested') or r.get('requirement_goal', '')).strip()}"]
        elif executive.get("what_was_requested"):
            out += [f"**Requested:** {str(executive.get('what_was_requested', '')).strip()}"]
        if executive.get("what_was_attempted"):
            out += [f"**Attempted:** {str(executive.get('what_was_attempted', '')).strip()}"]
        if executive.get("verdict_statement"):
            out += [f"**Verdict:** {str(executive.get('verdict_statement', '')).strip()}"]
        if executive.get("strongest_evidence"):
            out += [f"**Strongest evidence:** {str(executive.get('strongest_evidence', '')).strip()}"]
        if executive.get("rca_summary"):
            out += [f"**RCA summary:** {str(executive.get('rca_summary', '')).strip()}"]
        if executive.get("recommended_next_action"):
            out += [f"**Recommended next action:** {str(executive.get('recommended_next_action', '')).strip()}"]
        out += [""]

    if r.get("requirement_goal") or r.get("requirement_title"):
        out += ["## Requirement Context", ""]
        if r.get("requirement_title"):
            out += [f"**Title:** {str(r.get('requirement_title', '')).strip()}"]
        if r.get("requirement_goal"):
            out += [f"**Goal:** {str(r.get('requirement_goal', '')).strip()}"]
        out += [""]

    # Caveats come high up, not buried at the bottom. A reader who stops after
    # the summary should still see the limits of the result.
    caveats = r.get("caveats") or []
    if caveats:
        out += ["## What this result does NOT prove", ""]
        out += [f"- {c}" for c in caveats] + [""]

    health = r.get("health") or {}
    if health:
        out += ["## Rig health", "",
                "| Component | Status | Detail |", "|---|---|---|"]
        for c in health.get("components", []):
            status = "OK" if c.get("ok") else "FAIL"
            detail = str(c.get("detail", "")).replace("|", "\\|")
            out.append(f"| {c.get('name', '')} | {status} | {detail} |")
        out.append("")
        for issue in health.get("blocking_issues", []):
            out.append(f"- **Blocking:** {issue}")
        out.append("")

    verification = r.get("verification") or {}
    if verification:
        out += ["## Verification", ""]
        for c in verification.get("criteria", []):
            mark = "[x]" if c.get("met") else "[ ]"
            out.append(f"- {mark} **{c.get('criterion', '')}** — "
                       f"{c.get('reasoning', '')}")
        out.append("")
        for item in verification.get("not_proven", []):
            out.append(f"- _Not proven:_ {item}")
        out.append("")

    execution = r.get("execution") or {}
    steps = execution.get("steps") or []
    if steps:
        out += ["## Steps", "",
                "| # | Action | Dispatched | Screen delta | Observation |",
                "|---|---|---|---|---|"]
        for s in steps:
            delta = s.get("screen_delta")
            # "n/a" and 0.0 mean different things: not measured versus measured
            # and unchanged. The second is a finding; the first is a gap.
            delta_text = "n/a" if delta is None else f"{float(delta):.3f}"
            obs = str(s.get("observation", "")).replace("|", "\\|")[:120]
            out.append(
                f"| {s.get('index', '')} | `{s.get('action', '')}` | "
                f"{'yes' if s.get('dispatched') else 'no'} | {delta_text} | "
                f"{obs} |")
        out.append("")

    rca = r.get("rca") or {}
    if rca:
        out += [
            "## Root cause analysis", "",
            f"**Primary cause:** {rca.get('primary_cause', '')}",
            f"**Class:** `{rca.get('failure_class', '')}`  "
            f"**Severity:** `{rca.get('severity', '')}`  "
            f"**Confidence:** {rca.get('confidence', 0)}",
            "",
        ]
        hypotheses = rca.get("hypotheses") or []
        if hypotheses:
            out += ["### Hypotheses considered", ""]
            for h in hypotheses:
                out += [
                    f"**{h.get('statement', '')}** "
                    f"(`{h.get('failure_class', '')}`, "
                    f"likelihood {h.get('likelihood', 0)})",
                    "",
                ]
                for e in h.get("supporting_evidence", []):
                    out.append(f"  - supports: {e}")
                for e in h.get("contradicting_evidence", []):
                    out.append(f"  - against: {e}")
                out += [f"  - **next check:** {h.get('next_check', '')}", ""]
        recs = rca.get("recommendations") or []
        if recs:
            out += ["### Recommended actions", ""]
            out += [f"{i}. {rec}" for i, rec in enumerate(recs, 1)] + [""]

    # Screenshots, paired before/after per step so the change is visible side
    # by side rather than as an undifferentiated pile of images.
    steps = (r.get("execution") or {}).get("steps") or []
    paired = [s for s in steps if s.get("frame_before") or s.get("frame_after")]

    if paired:
        out += ["## Evidence", "",
                "Each step's before/after frames. The delta is the measured "
                "mean pixel difference - the actual proof that an action had "
                "an effect.", ""]
        for s in paired:
            delta = s.get("screen_delta")
            delta_text = "not measured" if delta is None else f"{float(delta):.3f}"
            out += [f"### Step {s.get('index')} — `{s.get('action')}` "
                    f"(delta {delta_text})", ""]
            if s.get("observation"):
                out += [f"_{s['observation']}_", ""]

            for label, key in (("Before", "frame_before"), ("After", "frame_after")):
                path = s.get(key)
                if not path:
                    continue
                # link_from_reports, NOT relative: this file lives in reports/
                # while the frames live in frames/, so the link needs "../".
                link = ctx.artifacts.link_from_reports(path)
                out += [f"**{label}:**", "", f"![{label} step {s.get('index')}]({link})", ""]

            if s.get("ocr_text"):
                out += ["<details><summary>Text read from this frame (OCR)</summary>",
                        "", "```", str(s["ocr_text"])[:1500], "```", "</details>", ""]

    elif r.get("screenshots"):
        out += ["## Evidence", ""]
        for shot in r["screenshots"]:
            link = ctx.artifacts.link_from_reports(shot)
            name = ctx.artifacts.relative(shot)
            out += [f"### {name}", "", f"![{name}]({link})", ""]

    out += ["---", "",
            "_Generated by the Xbox agentic testing framework. A PASS here "
            "means observed pixel evidence supported every success criterion; "
            "command acknowledgements alone never produce a PASS._"]
    return "\n".join(out)


# ===========================================================================
# HTML
# ===========================================================================
def _html(ctx: ToolContext, r: dict[str, Any]) -> str:
    verdict = str(r.get("verdict", "unknown")).upper()
    verdict_class = {
        "PASS": "pass",
        "FAIL": "fail",
        "BLOCKED": "blocked",
        "INCONCLUSIVE": "inconclusive",
        "ERROR": "error",
        "SKIPPED": "skipped",
    }.get(verdict, "unknown")

    executive = r.get("executive_summary") or {}
    health = r.get("health") or {}
    verification = r.get("verification") or {}
    execution = r.get("execution") or {}
    rca = r.get("rca") or {}
    steps = execution.get("steps") or []

    def p(text: Any) -> str:
        return html_escape(str(text or ""))

    def rel(path: Any) -> str:
        return ctx.artifacts.link_from_reports(str(path)) if path else ""

    sections: list[str] = []

    sections.append(f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{p(r.get('scenario_title') or r.get('scenario_id') or 'Test Report')}</title>
  <style>
    :root {{
      --bg: #0b1020;
      --panel: #121932;
      --panel-2: #182245;
      --text: #ecf2ff;
      --muted: #9eb0d1;
      --line: #2a3969;
      --accent: #74e0b8;
      --accent-2: #7cb7ff;
      --danger: #ff7f8f;
      --warn: #ffd36a;
      --blocked: #c8a2ff;
      --shadow: 0 20px 60px rgba(0,0,0,.28);
      --radius: 18px;
      --mono: "Cascadia Code", "SFMono-Regular", Consolas, monospace;
      --sans: "Segoe UI", "Inter", system-ui, sans-serif;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: var(--sans);
      background:
        radial-gradient(circle at top left, rgba(116,224,184,.14), transparent 32%),
        radial-gradient(circle at top right, rgba(124,183,255,.12), transparent 28%),
        linear-gradient(180deg, #08101d 0%, var(--bg) 100%);
      color: var(--text);
    }}
    .page {{ max-width: 1280px; margin: 0 auto; padding: 28px; }}
    .hero {{
      background: linear-gradient(135deg, rgba(18,25,50,.96), rgba(12,18,36,.96));
      border: 1px solid var(--line);
      border-radius: 26px;
      padding: 28px;
      box-shadow: var(--shadow);
      margin-bottom: 24px;
    }}
    .eyebrow {{
      display: inline-block;
      font-size: 12px;
      letter-spacing: .12em;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 10px;
    }}
    h1, h2, h3 {{ margin: 0 0 10px; }}
    h1 {{ font-size: 34px; line-height: 1.1; }}
    h2 {{ font-size: 22px; margin-bottom: 14px; }}
    h3 {{ font-size: 16px; margin-bottom: 10px; }}
    p, li {{ color: var(--text); line-height: 1.55; }}
    .muted {{ color: var(--muted); }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 14px;
      margin-top: 18px;
    }}
    .card {{
      background: rgba(18,25,50,.92);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      padding: 18px;
      box-shadow: var(--shadow);
    }}
    .metric {{
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: .08em;
      color: var(--muted);
      margin-bottom: 8px;
    }}
    .metric-value {{ font-size: 20px; font-weight: 700; }}
    .pill {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      border-radius: 999px;
      padding: 10px 14px;
      font-weight: 700;
      letter-spacing: .02em;
    }}
    .pill.pass {{ background: rgba(116,224,184,.14); color: var(--accent); }}
    .pill.fail, .pill.error {{ background: rgba(255,127,143,.14); color: var(--danger); }}
    .pill.blocked {{ background: rgba(200,162,255,.15); color: var(--blocked); }}
    .pill.inconclusive, .pill.skipped {{ background: rgba(255,211,106,.14); color: var(--warn); }}
    .section {{ margin-bottom: 22px; }}
    .cols {{
      display: grid;
      grid-template-columns: 1.15fr .85fr;
      gap: 18px;
    }}
    @media (max-width: 960px) {{ .cols {{ grid-template-columns: 1fr; }} }}
    .list {{ margin: 0; padding-left: 18px; }}
    .kv {{
      display: grid;
      grid-template-columns: 160px 1fr;
      gap: 8px 12px;
      font-size: 14px;
    }}
    .kv div:nth-child(odd) {{ color: var(--muted); }}
    table {{
      width: 100%;
      border-collapse: collapse;
      overflow: hidden;
      border-radius: 16px;
      border: 1px solid var(--line);
      background: rgba(14,21,41,.78);
    }}
    th, td {{
      text-align: left;
      padding: 12px 14px;
      border-bottom: 1px solid rgba(42,57,105,.55);
      vertical-align: top;
      font-size: 14px;
    }}
    th {{ color: var(--muted); font-weight: 600; background: rgba(24,34,69,.72); }}
    tr:last-child td {{ border-bottom: none; }}
    code {{
      font-family: var(--mono);
      background: rgba(255,255,255,.06);
      padding: 2px 6px;
      border-radius: 8px;
    }}
    .step {{
      background: rgba(18,25,50,.92);
      border: 1px solid var(--line);
      border-radius: 20px;
      padding: 18px;
      margin-bottom: 16px;
      box-shadow: var(--shadow);
    }}
    .step-head {{
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 16px;
      margin-bottom: 14px;
    }}
    .delta {{
      color: var(--accent-2);
      font-weight: 700;
      white-space: nowrap;
    }}
    .frames {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 14px;
      margin-top: 14px;
    }}
    figure {{
      margin: 0;
      background: rgba(10,15,31,.88);
      border: 1px solid var(--line);
      border-radius: 16px;
      overflow: hidden;
    }}
    figure img {{
      width: 100%;
      height: auto;
      display: block;
      background: #050913;
    }}
    figcaption {{
      padding: 10px 12px;
      color: var(--muted);
      font-size: 13px;
    }}
    details {{
      margin-top: 14px;
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 12px 14px;
      background: rgba(9,14,28,.74);
    }}
    summary {{ cursor: pointer; color: var(--accent-2); }}
    pre {{
      white-space: pre-wrap;
      word-break: break-word;
      font-family: var(--mono);
      color: #d7e4ff;
      margin: 10px 0 0;
      font-size: 13px;
    }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <div class="eyebrow">Executive AI Report</div>
      <h1>{p(r.get('scenario_title') or r.get('scenario_id') or 'Test Report')}</h1>
      <p class="muted">{p(r.get('summary', ''))}</p>
      <div class="grid">
        <div class="card"><div class="metric">Verdict</div><div class="metric-value"><span class="pill {verdict_class}">{p(verdict)}</span></div></div>
        <div class="card"><div class="metric">Run ID</div><div class="metric-value"><code>{p(r.get('run_id', ''))}</code></div></div>
        <div class="card"><div class="metric">Duration</div><div class="metric-value">{p(f"{float(r.get('duration_seconds', 0.0)):.1f}s")}</div></div>
        <div class="card"><div class="metric">Console</div><div class="metric-value">{p(r.get('console_profile', '') or 'default')}</div></div>
      </div>
    </section>
""")

    if executive:
        sections.append(f"""
    <section class="section card">
      <h2>Executive Summary</h2>
      <div class="kv">
        <div>Requested</div><div>{p(executive.get('what_was_requested', ''))}</div>
        <div>Attempted</div><div>{p(executive.get('what_was_attempted', ''))}</div>
        <div>Verdict</div><div>{p(executive.get('verdict_statement', ''))}</div>
        <div>Strongest evidence</div><div>{p(executive.get('strongest_evidence', ''))}</div>
        <div>RCA summary</div><div>{p(executive.get('rca_summary', ''))}</div>
        <div>Recommended next action</div><div>{p(executive.get('recommended_next_action', ''))}</div>
      </div>
    </section>
""")

    sections.append('<div class="cols">')

    sections.append('<div>')
    if r.get("requirement_id") or r.get("requirement_goal") or r.get("requirement_title"):
        sections.append(f"""
    <section class="section card">
      <h2>Requirement Context</h2>
      <div class="kv">
        <div>Requirement ID</div><div>{p(r.get('requirement_id', '') or '-')}</div>
        <div>Requirement Title</div><div>{p(r.get('requirement_title', '') or '-')}</div>
        <div>Requirement Goal</div><div>{p(r.get('requirement_goal', '') or '-')}</div>
      </div>
    </section>
""")

    caveats = r.get("caveats") or []
    if caveats:
        items = "".join(f"<li>{p(c)}</li>" for c in caveats)
        sections.append(f"""
    <section class="section card">
      <h2>What This Result Does Not Prove</h2>
      <ul class="list">{items}</ul>
    </section>
""")

    if verification:
        criteria = verification.get("criteria") or []
        crit_rows = "".join(
            f"<tr><td>{'PASS' if c.get('met') else 'FAIL'}</td><td>{p(c.get('criterion', ''))}</td><td>{p(c.get('reasoning', ''))}</td></tr>"
            for c in criteria
        )
        not_proven = verification.get("not_proven") or []
        not_proven_html = (
            "<ul class=\"list\">" +
            "".join(f"<li>{p(item)}</li>" for item in not_proven) +
            "</ul>"
        ) if not_proven else "<p class=\"muted\">None recorded.</p>"
        sections.append(f"""
    <section class="section card">
      <h2>Verification</h2>
      <table>
        <thead><tr><th>Status</th><th>Criterion</th><th>Reasoning</th></tr></thead>
        <tbody>{crit_rows or '<tr><td colspan="3">No verification criteria recorded.</td></tr>'}</tbody>
      </table>
      <h3 style="margin-top:16px;">Not proven</h3>
      {not_proven_html}
    </section>
""")

    if rca:
        recs = rca.get("recommendations") or []
        rec_html = "<ol class=\"list\">" + "".join(f"<li>{p(x)}</li>" for x in recs) + "</ol>" if recs else "<p class=\"muted\">No recommendations recorded.</p>"
        sections.append(f"""
    <section class="section card">
      <h2>Root Cause Analysis</h2>
      <div class="kv">
        <div>Primary cause</div><div>{p(rca.get('primary_cause', ''))}</div>
        <div>Failure class</div><div>{p(rca.get('failure_class', ''))}</div>
        <div>Severity</div><div>{p(rca.get('severity', ''))}</div>
        <div>Confidence</div><div>{p(rca.get('confidence', ''))}</div>
      </div>
      <h3 style="margin-top:16px;">Recommended actions</h3>
      {rec_html}
    </section>
""")
    sections.append('</div>')

    sections.append('<div>')
    if health:
        health_rows = "".join(
            f"<tr><td>{p(c.get('name', ''))}</td><td>{'OK' if c.get('ok') else 'FAIL'}</td><td>{p(c.get('detail', ''))}</td></tr>"
            for c in (health.get("components") or [])
        )
        sections.append(f"""
    <section class="section card">
      <h2>Rig & Console Information</h2>
      <div class="kv">
        <div>Console profile</div><div>{p(r.get('console_profile', '') or 'default')}</div>
        <div>Run finished</div><div>{p(r.get('finished_at', ''))}</div>
      </div>
      <h3 style="margin-top:16px;">Rig health</h3>
      <table>
        <thead><tr><th>Component</th><th>Status</th><th>Detail</th></tr></thead>
        <tbody>{health_rows or '<tr><td colspan="3">No health data recorded.</td></tr>'}</tbody>
      </table>
    </section>
""")

    metrics = r.get("metrics") or {}
    metric_html = "".join(
        f"<tr><td>{p(k)}</td><td>{p(v)}</td></tr>"
        for k, v in metrics.items()
    )
    sections.append(f"""
    <section class="section card">
      <h2>Run Metrics</h2>
      <table>
        <thead><tr><th>Metric</th><th>Value</th></tr></thead>
        <tbody>{metric_html or '<tr><td colspan="2">No metrics recorded.</td></tr>'}</tbody>
      </table>
    </section>
""")
    sections.append('</div>')
    sections.append('</div>')

    if steps:
        sections.append("""
    <section class="section">
      <h2>Step Timeline & Evidence</h2>
""")
        for s in steps:
            frames = []
            if s.get("frame_before"):
                frames.append(
                    f"<figure><img src=\"{p(rel(s.get('frame_before')))}\" alt=\"Before step {p(s.get('index', ''))}\"><figcaption>Before</figcaption></figure>")
            if s.get("frame_after"):
                frames.append(
                    f"<figure><img src=\"{p(rel(s.get('frame_after')))}\" alt=\"After step {p(s.get('index', ''))}\"><figcaption>After</figcaption></figure>")
            ocr = ""
            if s.get("ocr_text"):
                ocr = f"<details><summary>OCR text</summary><pre>{p(s.get('ocr_text', ''))}</pre></details>"
            sections.append(f"""
      <article class="step">
        <div class="step-head">
          <div>
            <h3>Step {p(s.get('index', ''))} — <code>{p(s.get('action', ''))}</code></h3>
            <p class="muted">{p(s.get('observation', ''))}</p>
          </div>
          <div class="delta">delta {p('n/a' if s.get('screen_delta') is None else f"{float(s.get('screen_delta', 0.0)):.3f}")}</div>
        </div>
        <div class="kv">
          <div>Dispatched</div><div>{'yes' if s.get('dispatched') else 'no'}</div>
          <div>Error</div><div>{p(s.get('error', '') or '-')}</div>
        </div>
        <div class="frames">{''.join(frames) or '<p class="muted">No frames saved for this step.</p>'}</div>
        {ocr}
      </article>
""")
        sections.append("    </section>")

    sections.append("""
  </div>
</body>
</html>
""")
    return "".join(sections)


# ===========================================================================
# JUnit
# ===========================================================================
def _junit(r: dict[str, Any]) -> str:
    verdict = str(r.get("verdict", "")).lower()
    name = r.get("scenario_title") or r.get("scenario_id") or "scenario"
    duration = float(r.get("duration_seconds", 0.0))
    summary = str(r.get("summary", ""))

    failures = 1 if verdict in ("fail", "error") else 0
    skipped = 1 if verdict in ("blocked", "skipped") else 0

    body = ""
    if verdict in ("fail", "error"):
        rca = r.get("rca") or {}
        detail = rca.get("primary_cause", summary) or summary
        body = (f'      <failure message={quoteattr(str(detail)[:300])} '
                f'type={quoteattr(str(rca.get("failure_class", "failure")))}>'
                f'{escape(summary)}</failure>\n')
    elif verdict in ("blocked", "skipped"):
        health = r.get("health") or {}
        reason = "; ".join(health.get("blocking_issues", [])) or summary
        body = f'      <skipped message={quoteattr(reason[:300])}/>\n'
    elif verdict == "inconclusive":
        # Not a pass. An inconclusive run that quietly went green would be the
        # exact false-pass this framework exists to prevent.
        body = (f'      <failure message={quoteattr("Inconclusive: " + summary[:280])} '
                f'type="inconclusive"/>\n')

    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<testsuites tests="1" failures="{failures}" skipped="{skipped}" '
        f'time="{duration:.3f}">\n'
        f'  <testsuite name="console-agentic-tests" tests="1" '
        f'failures="{failures}" skipped="{skipped}" time="{duration:.3f}">\n'
        f'    <testcase classname="agentic.console" name={quoteattr(str(name))} '
        f'time="{duration:.3f}">\n'
        f'{body}'
        f'      <system-out>{escape(json.dumps(r.get("metrics", {}), default=str))}'
        f'</system-out>\n'
        '    </testcase>\n'
        '  </testsuite>\n'
        '</testsuites>\n'
    )


def provide() -> list[ToolSpec]:
    return [
        ToolSpec("write_json_report", "Write the JSON run record.",
                 ["report"], _write_json_report),
        ToolSpec("write_markdown_report", "Write the human Markdown report.",
                 ["report"], _write_markdown_report),
        ToolSpec("write_html_report", "Write the styled HTML executive report.",
                 ["report"], _write_html_report),
        ToolSpec("write_junit_report", "Write JUnit XML for CI.",
                 ["report"], _write_junit_report),
        ToolSpec("list_artifacts", "List all files produced this run.",
                 ["report", "analysis"], _list_artifacts),
    ]
