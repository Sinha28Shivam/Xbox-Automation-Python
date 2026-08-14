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
        ToolSpec("write_junit_report", "Write JUnit XML for CI.",
                 ["report"], _write_junit_report),
        ToolSpec("list_artifacts", "List all files produced this run.",
                 ["report", "analysis"], _list_artifacts),
    ]
