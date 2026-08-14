"""
executor_agent.py - agent 4: drive the console and record what happened.

The only agent allowed to touch hardware. It walks the plan and, for every
step, runs the same loop:

    capture BEFORE  ->  act  ->  wait for the screen to settle  ->  capture AFTER

That before/after pair is the whole reason this framework can be trusted. It is
captured mechanically, for every step, whether or not the step "succeeded",
because evidence gathered only when things look good is not evidence.

WHAT IT DELIBERATELY DOES NOT DO
--------------------------------
It never decides pass or fail. It reports what it did and what it saw; the
verifier judges. Letting the actor grade its own work is how optimistic agents
produce green runs from nothing, so the executor's schema has no verdict field
at all - the separation is structural, not a matter of instruction.

EARLY ABORT
-----------
If several consecutive steps produce zero screen change, execution stops. That
pattern almost always means input is not reaching the console (an
unauthenticated GIMX session being the classic cause), and hammering forty more
buttons at a console that is not listening wastes minutes and muddies the RCA
timeline. Stopping early with a clear signal is more useful than finishing.
"""

from __future__ import annotations

import time
from typing import Any

from base import BaseAgent
from schemas import (
    Evidence,
    EvidenceKind,
    ExecutionResult,
    PlannedStep,
    StepResult,
)
from state import AgenticState, note


class ExecutorAgent(BaseAgent):
    """Runs the plan against real hardware, capturing evidence throughout."""

    role = "executor"
    uses_llm = False

    def run(self, state: AgenticState) -> dict[str, Any]:
        plan = state.get("plan")
        scenario = state.get("scenario")
        if plan is None or scenario is None:
            raise ValueError("Executor ran without a plan or scenario.")

        dry_run = bool(state.get("dry_run", False))
        max_steps = int(self.context.settings.get("runtime.max_steps", 40))
        run_deadline = time.time() + float(
            scenario.timeout_seconds
            or self.context.settings.get("runtime.run_timeout_seconds", 900))

        results: list[StepResult] = []
        notes: list[str] = []
        aborted = False
        abort_reason: str | None = None
        unchanged_streak = 0
        started = time.time()

        for step in plan.steps[:max_steps]:
            if time.time() > run_deadline:
                aborted, abort_reason = True, (
                    f"Run timeout exceeded after {len(results)} steps.")
                break

            result = self._execute_step(step, dry_run)
            results.append(result)

            # Only a HARD failure aborts. A step that merely observed nothing
            # useful is recorded and the run continues - previously a single
            # bad argument name killed a 6-step plan at step 2, throwing away
            # the remaining evidence and leaving the console mid-transition.
            if result.error and not step.optional and self._is_fatal(result):
                aborted, abort_reason = True, (
                    f"Step {step.index} ({step.action}) failed: {result.error}")
                break

            if result.error:
                notes.append(
                    f"Step {step.index} ({step.action}) reported an error but "
                    f"the run continued: {result.error}")

            # Track consecutive no-change steps. A single one is normal (a
            # deliberate wait, a press on an already-selected item); three in a
            # row means nothing is getting through.
            if result.screen_delta is not None:
                threshold = self.context.settings.get(
                    "verification.screen_change_threshold", 0.5)
                unchanged_streak = (0 if result.screen_delta >= threshold
                                    else unchanged_streak + 1)

            if unchanged_streak >= 3 and not dry_run:
                aborted = True
                abort_reason = (
                    f"The screen did not change across {unchanged_streak} "
                    f"consecutive actions. Input is almost certainly not "
                    f"reaching the console - the usual cause is a GIMX session "
                    f"that was never authenticated with the Guide button. "
                    f"Stopping rather than sending more input into a void.")
                notes.append(abort_reason)
                break

        execution = ExecutionResult(
            scenario_id=scenario.id,
            completed=not aborted and len(results) == len(plan.steps),
            aborted=aborted,
            abort_reason=abort_reason,
            steps=results,
            total_steps=len(plan.steps),
            dispatched_steps=sum(1 for r in results if r.dispatched),
            duration_seconds=round(time.time() - started, 2),
            action_log=self._action_log(),
            notes=notes,
        )

        self.context.artifacts.save_json(
            "execution.json", execution.model_dump(mode="json"))

        summary = (
            f"Executed {len(results)}/{len(plan.steps)} steps; "
            f"{execution.dispatched_steps} dispatched; "
            f"screen changed at least once: {execution.observed_any_change}."
            + (f" ABORTED: {abort_reason}" if aborted else ""))

        return {
            "execution": execution,
            "step_count": int(state.get("step_count", 0)) + len(results),
            "messages": [note(self.role, summary,
                              level="error" if aborted else "info")],
            "agent_outputs": {self.role: {
                "ok": True,
                "aborted": aborted,
                "steps_run": len(results),
                "observed_any_change": execution.observed_any_change,
            }},
        }

    # -- one step ----------------------------------------------------------
    def _execute_step(self, step: PlannedStep, dry_run: bool) -> StepResult:
        started = time.time()
        result = StepResult(index=step.index, action=step.action,
                            arguments=dict(step.arguments))

        before_path = self._capture(f"{step.index}-before-{step.action}",
                                    step.index, dry_run)
        result.frame_before = before_path

        try:
            tool_result = self._dispatch(step, dry_run)
        except Exception as exc:
            result.error = f"{exc.__class__.__name__}: {exc}"
            result.duration_seconds = round(time.time() - started, 2)
            return result

        result.dispatched = bool(tool_result.get("dispatched",
                                                 tool_result.get("ok", False)))
        if not tool_result.get("ok"):
            result.error = str(tool_result.get("error", "tool reported failure"))

        # Recorded as evidence, but with COMMAND_ACK - a kind the schema
        # excludes from proof. The acknowledgement belongs in the timeline; it
        # just must never be mistaken for an observation.
        result.evidence.append(Evidence(
            kind=EvidenceKind.COMMAND_ACK,
            summary=f"{step.action} dispatched={result.dispatched}",
            detail=_jsonable(tool_result),
            source_tool=step.action,
        ))

        if step.verify and not dry_run:
            self._observe(step, result)
            self._read_text(result)

        result.duration_seconds = round(time.time() - started, 2)
        return result

    @staticmethod
    def _is_fatal(result: StepResult) -> bool:
        """Should this error stop the whole run?

        Fatal: the hardware layer is gone (capture device lost, controller
        unavailable). Continuing would produce nothing but more errors.

        Not fatal: an unknown tool, a bad argument, a failed OCR read. Those
        cost one step's evidence, not the run's. Aborting on them discards the
        frames we already captured - which are often enough to reach a verdict.
        """
        text = (result.error or "").lower()
        fatal_signs = ("capture unavailable", "controller unavailable",
                       "adapter", "device may have been taken")
        return any(sign in text for sign in fatal_signs)

    def _read_text(self, result: StepResult) -> None:
        """OCR the 'after' frame so text criteria can be checked later.

        Done for EVERY verified step, not only when a plan asks for it. The
        verifier frequently needs to answer "was an error dialog showing?"
        about a step nobody thought to OCR at planning time, and the frame is
        already on disk - reading it costs far less than an inconclusive run.
        """
        if not result.frame_after:
            return
        if not self.context.settings.get("verification.ocr.enabled", True):
            return

        reading = self.call_tool("read_screen_text", frame_path=result.frame_after)
        if not reading.get("ok"):
            return

        text = str(reading.get("text", "")).strip()
        if not text:
            return

        result.ocr_text = text
        result.ocr_engine = reading.get("engine")

        lines = [l.strip() for l in text.splitlines() if l.strip()]
        result.evidence.append(Evidence(
            kind=EvidenceKind.OCR_TEXT,
            summary=f"Read {len(lines)} line(s) via {reading.get('engine')}: "
                    f"{'; '.join(lines[:4])[:200]}",
            detail={"text": text[:4000], "engine": reading.get("engine")},
            frame_path=result.frame_after,
            # OCR on console UIs is unreliable, so this is proof-grade evidence
            # but never full confidence - the verifier weighs it accordingly.
            confidence=0.75,
            source_tool="read_screen_text",
        ))

    def _dispatch(self, step: PlannedStep, dry_run: bool) -> dict[str, Any]:
        if dry_run:
            return {"ok": True, "dispatched": False, "dry_run": True}

        available = {s["name"] for s in self.tool_catalogue()}
        if step.action not in available:
            return {
                "ok": False,
                "dispatched": False,
                "error": (f"'{step.action}' is not a tool this agent may call. "
                          f"Available: {', '.join(sorted(available))}"),
            }
        return self.call_tool(step.action, **step.arguments)

    def _observe(self, step: PlannedStep, result: StepResult) -> None:
        """Wait for the UI to settle, capture, and measure the change."""
        stable = self.call_tool(
            "wait_for_stable_screen",
            timeout=step.timeout_seconds,
            label=f"{step.index}-after-{step.action}")

        after_path = (stable.get("frame_path") if stable.get("ok")
                      else self._capture(f"{step.index}-after-{step.action}",
                                         step.index, False))
        result.frame_after = after_path

        if not (result.frame_before and after_path):
            result.observation = (
                "No frames were captured, so this step's effect is unknown. "
                "That is a gap in the evidence, not a pass.")
            return

        comparison = self.call_tool(
            "verify_screen_changed",
            before_path=result.frame_before,
            after_path=after_path)

        if not comparison.get("ok"):
            result.observation = f"Comparison failed: {comparison.get('error')}"
            return

        delta = float(comparison.get("delta", 0.0))
        changed = bool(comparison.get("changed"))
        result.screen_delta = delta
        result.observation = (
            f"Screen {'changed' if changed else 'did NOT change'} "
            f"(delta {delta:.3f}). Expected: {step.expected_observation}")

        # Only recorded as SCREEN_DIFF proof when the screen genuinely moved.
        # A no-change result is logged as a frame statistic instead, so it can
        # never accidentally support a PASS.
        result.evidence.append(Evidence(
            kind=EvidenceKind.SCREEN_DIFF if changed else EvidenceKind.FRAME_STATS,
            summary=result.observation,
            detail=_jsonable(comparison),
            frame_path=after_path,
            source_tool="verify_screen_changed",
        ))

    # -- helpers -----------------------------------------------------------
    def _capture(self, label: str, step: int, dry_run: bool) -> str | None:
        if dry_run:
            return None
        result = self.call_tool("capture_frame", label=label, step=step)
        return result.get("frame_path") if result.get("ok") else None

    def _action_log(self) -> list[dict[str, Any]]:
        """ConsolePad's own log: every raw GIMX event with its status.

        Lower level than our step results and very useful for RCA - it shows
        retries and per-event failures that a step-level view hides.
        """
        try:
            return list(self.context.tools.context.hardware.pad().action_log)
        except Exception:
            return []


def _jsonable(payload: Any) -> dict[str, Any]:
    """Keep only JSON-safe values - tool results can carry odd objects."""
    if not isinstance(payload, dict):
        return {"value": str(payload)}
    out: dict[str, Any] = {}
    for k, v in payload.items():
        out[k] = v if isinstance(v, (str, int, float, bool, type(None),
                                     list, dict)) else str(v)
    return out
