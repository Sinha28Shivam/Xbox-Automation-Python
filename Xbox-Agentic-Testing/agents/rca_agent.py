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

from base import BaseAgent
from schemas import FailureClass, Hypothesis, RootCauseAnalysis, Severity
from state import AgenticState, note


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

        summary = (f"Root cause ({rca.failure_class.value}, "
                   f"confidence {rca.confidence:.0%}): {rca.primary_cause}")

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

        if rca.severity == Severity.CRITICAL and rca.confidence < 0.5:
            # A confident-sounding critical claim on weak evidence is worse
            # than an honest "we are not sure yet".
            rca.severity = Severity.HIGH
            rca.ruled_out.append(
                "Severity reduced from critical: confidence was below 50%.")

        return rca

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
