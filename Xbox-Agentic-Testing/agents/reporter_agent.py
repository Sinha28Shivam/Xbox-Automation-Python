"""
reporter_agent.py - agent 7: produce the artefact humans and CI read.

Assembles everything into a TestReport and writes it in whichever formats are
configured. It never re-judges anything: the verdict comes from the verifier
(or from health, when the run was blocked). A reporter that could revise the
verdict would be a second, unaccountable judge.

DETERMINISTIC
-------------
No LLM. Every summary here is assembled from data that already exists, and the
verdict resolution is a handful of explicit rules. Asking a model to restate
facts it was just given adds cost and a chance of drift to a step that has none.

THE CAVEATS SECTION
-------------------
Generated mechanically from what actually happened, and placed near the top of
the Markdown. A PASS backed only by "the screen changed" is much weaker than one
backed by reading the expected text, and the report should say so rather than
letting a green heading imply more than the evidence supports.
"""

from __future__ import annotations

import time
from typing import Any

from base import BaseAgent
from schemas import ExecutiveSummary, TestReport, Verdict
from state import AgenticState, note


class ReporterAgent(BaseAgent):
    """Writes the final report in every configured format."""

    role = "reporter"

    def run(self, state: AgenticState) -> dict[str, Any]:
        requirement = state.get("requirement")
        scenario = state.get("scenario")
        health = state.get("health")
        execution = state.get("execution")
        verification = state.get("verification")
        rca = state.get("rca")

        verdict = self._resolve_verdict(state)
        started_at = str(state.get("started_at", ""))
        executive_summary = self._executive_summary(state, verdict)

        report = TestReport(
            run_id=str(state.get("run_id", "")),
            scenario_id=scenario.id if scenario else "unknown",
            scenario_title=scenario.title if scenario else "Unknown scenario",
            console_profile=(str(scenario.console or "") if scenario else ""),
            requirement_id=(requirement.id if requirement else None),
            requirement_title=(requirement.title if requirement else ""),
            requirement_goal=(requirement.goal if requirement else ""),
            verdict=verdict,
            summary=self._summary(verdict, state),
            executive_summary=executive_summary,
            started_at=started_at,
            duration_seconds=self._duration(started_at),
            health=health,
            plan=state.get("plan"),
            execution=execution,
            verification=verification,
            rca=rca,
            artifacts=list(self.context.artifacts.files),
            screenshots=self.context.artifacts.list_frames(),
            caveats=self._caveats(state, verdict),
            metrics=self._metrics(state),
            stage_summary=(execution.stage_summary if execution else []),
            stage_transitions=(execution.stage_transitions if execution else []),
        )

        payload = report.model_dump(mode="json")
        written = self._write(payload)
        report.report_files = written

        self.context.artifacts.save_json(
            "state-digest.json",
            {"messages": state.get("messages", []),
             "errors": state.get("errors", []),
             "agent_outputs": state.get("agent_outputs", {})},
            subdir="logs")

        return {
            "report": report,
            "messages": [note(
                self.role,
                f"{verdict.value.upper()} - report written to "
                f"{self.context.artifacts.reports_dir}")],
            "should_stop": True,
            "stop_reason": "Report complete",
            "agent_outputs": {self.role: {
                "ok": True,
                "verdict": verdict.value,
                "files": written,
            }},
        }

    def _executive_summary(self, state: AgenticState,
                           verdict: Verdict) -> ExecutiveSummary:
        """Generate an LLM-written narrative without changing any verdict data."""
        scenario = state.get("scenario")
        requirement = state.get("requirement")
        verification = state.get("verification")
        rca = state.get("rca")
        execution = state.get("execution")
        health = state.get("health")

        fallback = ExecutiveSummary(
            what_was_requested=(
                requirement.goal if requirement else
                (scenario.goal if scenario else "Unknown request")
            ),
            what_was_attempted=(
                f"Executed {len(execution.steps) if execution else 0} planned steps "
                f"through the current pipeline."
            ),
            verdict_statement=(
                verification.summary if verification else
                f"Run ended with verdict {verdict.value}."
            ),
            strongest_evidence=(
                rca.primary_cause if rca else
                (verification.summary if verification else "No summary available.")
            ),
            rca_summary=(
                rca.primary_cause if rca else
                ("Rig was blocked before product behaviour could be judged."
                 if health and not health.healthy else "No RCA was produced.")
            ),
            recommended_next_action=(
                rca.recommendations[0] if rca and rca.recommendations else
                ("Fix the blocking rig issue and re-run."
                 if health and not health.healthy else
                 "Review the captured evidence and re-run if needed.")
            ),
        )

        try:
            prompt = self.render_prompt(
                state,
                verdict=verdict.value,
                scenario=(scenario.model_dump(mode="json") if scenario else None),
                requirement=(requirement.model_dump(mode="json")
                             if requirement else None),
                health=(health.model_dump(mode="json") if health else None),
                execution=(execution.model_dump(mode="json") if execution else None),
                verification=(verification.model_dump(mode="json")
                              if verification else None),
                rca=(rca.model_dump(mode="json") if rca else None),
            )
            return self.invoke_structured(ExecutiveSummary, prompt)
        except Exception as exc:
            self.context.artifacts.append_log(
                "reporter.log",
                f"Executive summary fallback used: {exc}")
            return fallback

    # -- verdict -----------------------------------------------------------
    @staticmethod
    def _resolve_verdict(state: AgenticState) -> Verdict:
        """Pick the run's verdict. Order of precedence matters.

        BLOCKED wins over everything: if the rig was unusable, no statement
        about the console is supportable, and reporting FAIL would blame the
        product for our own broken equipment.
        """
        health = state.get("health")
        if health is not None and not health.healthy:
            return Verdict.BLOCKED

        verification = state.get("verification")
        if verification is not None:
            return verification.verdict

        if state.get("errors"):
            return Verdict.ERROR

        # Reached the reporter with no verdict at all - a routing bug. Say so
        # rather than defaulting to something that looks like a real result.
        return Verdict.INCONCLUSIVE

    @staticmethod
    def _summary(verdict: Verdict, state: AgenticState) -> str:
        scenario = state.get("scenario")
        title = scenario.title if scenario else "the scenario"

        if verdict == Verdict.BLOCKED:
            health = state.get("health")
            issues = "; ".join(health.blocking_issues) if health else "unknown"
            return (f"BLOCKED - '{title}' was never run because the rig was "
                    f"not usable: {issues}. This says nothing about the "
                    f"console; fix the rig and re-run.")

        verification = state.get("verification")
        if verification is None:
            return (f"No verdict was produced for '{title}'. This indicates a "
                    f"framework problem, not a console result.")

        rca = state.get("rca")
        base = f"{verdict.value.upper()} - {verification.summary}"
        if rca is not None:
            base += f" Root cause ({rca.failure_class.value}): {rca.primary_cause}"
        return base

    # -- caveats -----------------------------------------------------------
    @staticmethod
    def _caveats(state: AgenticState, verdict: Verdict) -> list[str]:
        """State plainly what this result does not establish."""
        caveats: list[str] = []
        health = state.get("health")
        execution = state.get("execution")
        verification = state.get("verification")

        if state.get("dry_run"):
            caveats.append(
                "DRY RUN: no input was sent and no frames were captured. "
                "Nothing about console behaviour was tested.")

        if health is not None:
            caveats.extend(health.warnings)
            if health.gimx_authenticated is None and verdict != Verdict.BLOCKED:
                caveats.append(
                    "GIMX authentication was never directly confirmed. It is "
                    "inferred only from the screen having changed.")

        if verification is not None:
            caveats.extend(verification.not_proven)
            if verification.last_proven_stage is not None:
                caveats.append(
                    f"Last proven stage: {verification.last_proven_stage.value}.")

        if execution is not None:
            unverified = [s.index for s in execution.steps
                          if s.screen_delta is None]
            if unverified:
                caveats.append(
                    f"Steps {unverified} produced no frame comparison, so "
                    f"their effect is unknown rather than confirmed.")
            if execution.total_steps > len(execution.steps):
                caveats.append(
                    f"Only {len(execution.steps)} of {execution.total_steps} "
                    f"planned steps ran; the rest were never attempted.")

        if verdict == Verdict.PASS:
            caveats.append(
                "A PASS means the observed evidence supported every success "
                "criterion. It does not prove the absence of defects that this "
                "scenario did not look for.")
        return caveats

    # -- metrics -----------------------------------------------------------
    @staticmethod
    def _metrics(state: AgenticState) -> dict[str, Any]:
        execution = state.get("execution")
        deltas = ([s.screen_delta for s in execution.steps
                   if s.screen_delta is not None] if execution else [])
        return {
            "steps_planned": execution.total_steps if execution else 0,
            "steps_run": len(execution.steps) if execution else 0,
            "steps_dispatched": execution.dispatched_steps if execution else 0,
            "steps_with_screen_change": sum(1 for d in deltas if d > 0),
            "mean_screen_delta": (round(sum(deltas) / len(deltas), 4)
                                  if deltas else 0.0),
            "replans": int(state.get("replan_count", 0)),
            "errors": len(state.get("errors", [])),
            "execution_seconds": execution.duration_seconds if execution else 0.0,
        }

    @staticmethod
    def _duration(started_at: str) -> float:
        if not started_at:
            return 0.0
        try:
            from datetime import datetime, timezone
            start = datetime.fromisoformat(started_at)
            if start.tzinfo is None:
                start = start.replace(tzinfo=timezone.utc)
            return round((datetime.now(timezone.utc) - start).total_seconds(), 2)
        except (ValueError, TypeError):
            return 0.0

    # -- writing -----------------------------------------------------------
    def _write(self, payload: dict[str, Any]) -> dict[str, str]:
        """Write every configured format, tolerating individual failures.

        One broken writer must not cost the whole report - losing the Markdown
        because the JUnit writer choked would be a poor trade.
        """
        writers = {
            "json": ("write_json_report", "report.json"),
            "markdown": ("write_markdown_report", "report.md"),
            "html": ("write_html_report", "report.html"),
            "junit": ("write_junit_report", "junit.xml"),
        }
        written: dict[str, str] = {}
        for fmt in self.context.settings.list_of("reporting.formats"):
            entry = writers.get(str(fmt))
            if entry is None:
                continue
            tool_name, filename = entry
            result = self.call_tool(tool_name, report=payload, filename=filename)
            if result.get("ok"):
                written[str(fmt)] = result["path"]
            else:
                self.context.artifacts.append_log(
                    "reporter.log",
                    f"Failed to write {fmt}: {result.get('error')}")
        return written
