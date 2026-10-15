"""
rca_agent.py - agent 6: why did it fail?

Runs only on failure. Its job is the distinction that matters most in console
testing:

    Is the RIG broken, or is the CONSOLE broken?

Getting that wrong is expensive in both directions. Blame the console and
someone files a bug against a healthy product. Blame the rig and a real defect
ships. So the agent is required to produce ranked hypotheses, each with the
evidence for and against it and - crucially - a concrete next check. A
hypothesis nobody can test is just an opinion with a confidence score.

THE GUARDRAIL
-------------
`product_bug_suspected` is forced to False whenever the health report showed
problems, no matter what the model concluded. You cannot blame the console on
evidence gathered through a broken measuring instrument. That is enforced in
code below rather than requested in the prompt.
"""

from __future__ import annotations

from typing import Any

from agents.base import BaseAgent
from core.lesson_store import record_failure
from core.schemas import FailureClass, Hypothesis, RootCauseAnalysis, Severity
from core.state import AgenticState, note


class RCAAgent(BaseAgent):
    """Explains a failure and classifies where the fault lies."""

    role = "rca"

    def run(self, state: AgenticState) -> dict[str, Any]:
        scenario = state.get("scenario")
        health = state.get("health")
        execution = state.get("execution")
        verification = state.get("verification")

        scenario_id = scenario.id if scenario else "unknown"

        # Re-probe the rig NOW rather than trusting the pre-run snapshot.
        # Hardware can fail mid-run: a USB drop or an app stealing the capture
        # device halfway through looks exactly like a console fault otherwise.
        current_health = {
            "gimx": self.call_tool("check_gimx_session"),
            "capture": self.call_tool("check_capture_device"),
        }

        prompt = self.render_prompt(
            state,
            scenario=scenario.model_dump(mode="json") if scenario else None,
            health_before=health.model_dump(mode="json") if health else None,
            health_now=current_health,
            execution=execution.model_dump(mode="json") if execution else None,
            verification=(verification.model_dump(mode="json")
                          if verification else None),
            timeline=self._timeline(state),
            action_log=(execution.action_log[-40:] if execution else []),
            failure_classes=[c.value for c in FailureClass],
        )

        rca = self.invoke_structured(RootCauseAnalysis, prompt)
        rca.scenario_id = scenario_id
        rca = self._apply_guardrails(rca, state, current_health)

        self.context.artifacts.save_json(
            "rca.json", rca.model_dump(mode="json"))

        lesson = None
        if not state.get("dry_run"):
            plan = state.get("plan")
            lesson = record_failure(
                self.context.artifacts.run_dir, scenario_id,
                rca=rca, execution=execution, verification=verification,
                run_id=str(state.get("run_id", "")),
                revision=int(getattr(plan, "revision", 1) or 1),
                run_dir=self.context.artifacts.run_dir,
                tags=getattr(scenario, "tags", None),
                shared_tags=self.context.settings.list_of("learning.shared_tags") or None)
            if lesson:
                self.context.artifacts.save_json(
                    f"lesson-recorded-r{getattr(plan, 'revision', 1)}.json", lesson)

        summary = (f"Root cause ({rca.failure_class.value}, "
                   f"confidence {rca.confidence:.0%}): {rca.primary_cause}"
                   + (f" | lesson {lesson['id']} recorded "
                      f"(seen {lesson.get('occurrences', 1)}x)" if lesson else ""))

        return {
            "rca": rca,
            "messages": [note(self.role, summary, level="error")],
            "agent_outputs": {self.role: {
                "ok": True,
                "failure_class": rca.failure_class.value,
                "retryable": rca.is_retryable,
                "product_bug_suspected": rca.product_bug_suspected,
            }},
        }

    # -- guardrails --------------------------------------------------------
    def _apply_guardrails(self, rca: RootCauseAnalysis, state: AgenticState,
                          current_health: dict[str, Any]) -> RootCauseAnalysis:
        """Correct conclusions the evidence cannot support."""
        health = state.get("health")
        execution = state.get("execution")

        rig_was_unhealthy = bool(health and not health.healthy)
        rig_now_unhealthy = not (
            current_health.get("gimx", {}).get("reachable")
            and current_health.get("capture", {}).get("has_signal"))

        if rig_was_unhealthy or rig_now_unhealthy:
            if rca.product_bug_suspected:
                rca.product_bug_suspected = False
                rca.ruled_out.append(
                    "Product defect cannot be concluded: the test rig itself "
                    "was unhealthy, so any observation of the console was made "
                    "through a broken instrument.")
            if rca.failure_class == FailureClass.PRODUCT_DEFECT:
                rca.failure_class = FailureClass.RIG_FAULT

        # The signature failure of this whole project. When nothing ever
        # appeared on screen, the leading hypothesis is always the same, and
        # it is worth stating explicitly rather than hoping the model found it.
        if execution is not None and not execution.observed_any_change:
            rca.product_bug_suspected = False
            if not any("guide" in h.statement.lower() for h in rca.hypotheses):
                rca.hypotheses.insert(0, Hypothesis(
                    statement=(
                        "The GIMX session was never authenticated: nobody held "
                        "the Guide button for 2 seconds. Events are accepted "
                        "and acknowledged, but nothing reaches the console."),
                    failure_class=FailureClass.RIG_FAULT,
                    likelihood=0.7,
                    supporting_evidence=[
                        "The screen never changed across the entire run.",
                        "Events were dispatched successfully, so GIMX itself "
                        "was reachable and accepting input.",
                    ],
                    next_check=(
                        "Hold the controller's GUIDE button for 2 seconds, "
                        "then press a D-pad direction and watch the TV. If the "
                        "selection moves, this was the cause."),
                ))
            rca.recommendations.insert(
                0, "Re-authenticate the GIMX session (hold Guide for 2s) and "
                   "confirm on the TV that input moves the selection.")

        # An abort raised by our OWN safety guard is an automation outcome by
        # definition: the console did what it was told, and the framework then
        # declined to continue. Twice (run-20261014-162340, run-20261015-113636)
        # RCA filed this as product_defect, which would raise a bug against a
        # healthy console - and would also stop the lesson steering the plan.
        abort = (execution.abort_reason or "") if execution is not None else ""
        if rca.failure_class == FailureClass.PRODUCT_DEFECT and any(
                marker in abort for marker in (
                    "Blind tapping prevented", "Selection blocked",
                    "Run timeout exceeded")):
            rca.failure_class = FailureClass.AUTOMATION_DEFECT
            rca.product_bug_suspected = False
            rca.ruled_out.append(
                "Product defect ruled out: the run was stopped by the "
                f"framework's own guard ('{abort[:120]}'), not by the console "
                "misbehaving.")

        # A product defect is a claim about the CONSOLE, so it needs console
        # evidence: an error dialog, or the screen not responding to input
        # that was delivered. Without either, the likelier story is that our
        # plan asked for the wrong thing - and filing a console bug on that
        # basis is the expensive mistake this agent exists to prevent.
        if rca.failure_class == FailureClass.PRODUCT_DEFECT:
            evidence = self._product_defect_evidence(state)
            if evidence:
                rca.ruled_out.append(f"Product-defect evidence: {evidence}")
            else:
                rca.failure_class = FailureClass.AUTOMATION_DEFECT
                rca.product_bug_suspected = False
                rca.ruled_out.append(
                    "Product defect not supported: no error dialog was captured "
                    "and the console responded to input, so the plan, not the "
                    "console, is the likelier cause.")

        if rca.severity == Severity.CRITICAL and rca.confidence < 0.5:
            # A confident-sounding critical claim on weak evidence is worse
            # than an honest "we are not sure yet".
            rca.severity = Severity.HIGH
            rca.ruled_out.append(
                "Severity reduced from critical: confidence was below 50%.")

        return rca

    _ERROR_DIALOG = ("something went wrong", "error code", "0x8", "try again later",
                     "isn't working", "couldn't", "can't connect", "unexpected error")

    @classmethod
    def _product_defect_evidence(cls, state: AgenticState) -> str:
        """Observed console misbehaviour that could justify a product defect."""
        execution = state.get("execution")
        if execution is None:
            return ""
        for step in execution.steps:
            text = (step.ocr_text or "").lower()
            hit = next((p for p in cls._ERROR_DIALOG if p in text), None)
            if hit:
                return f"error text '{hit}' on screen after step {step.index}"
        # Delivered input that the console ignored, while other input worked.
        inputs = [s for s in execution.steps
                  if s.dispatched and s.action in {"press_button", "hold_button",
                                                   "move_stick", "run_sequence"}
                  and s.screen_delta is not None]
        trailing = 0
        for s in reversed(inputs):
            if s.screen_delta >= 0.05:
                break
            trailing += 1
        responsive_before = any(s.screen_delta >= 0.05 for s in inputs[:len(inputs) - trailing])
        if trailing >= 3 and responsive_before:
            return (f"the last {trailing} delivered inputs produced no screen "
                    f"change after the console had been responsive")
        return ""

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _timeline(state: AgenticState) -> list[str]:
        """A readable, ordered account of the run for the prompt."""
        lines: list[str] = []
        health = state.get("health")
        if health:
            lines.append(f"HEALTH: {health.summary}")
            lines += [f"  blocking: {b}" for b in health.blocking_issues]
            lines += [f"  warning: {w}" for w in health.warnings]

        execution = state.get("execution")
        if execution:
            for step in execution.steps:
                delta = ("n/a" if step.screen_delta is None
                         else f"{step.screen_delta:.3f}")
                lines.append(
                    f"STEP {step.index} {step.action} "
                    f"dispatched={step.dispatched} delta={delta} "
                    f"{step.observation or step.error or ''}".strip())
            if execution.abort_reason:
                lines.append(f"ABORT: {execution.abort_reason}")

        verification = state.get("verification")
        if verification:
            lines.append(
                f"VERDICT: {verification.verdict.value} - {verification.summary}")
        return lines
