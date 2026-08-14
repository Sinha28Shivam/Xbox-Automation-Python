"""
verifier_agent.py - agent 5: the anti-false-pass gate.

Deliberately a different agent from the executor. An agent that both acts and
grades its own actions will, sooner or later, decide it did well. This one sees
only the evidence - frames, deltas, OCR - and never learns what the executor
thought of its own performance.

THREE LAYERS OF DEFENCE AGAINST A FALSE PASS
--------------------------------------------
1. A deterministic pre-check. If the screen never changed during the entire
   run, no amount of model reasoning can turn that into a pass, so we return
   FAIL without spending a call.
2. The LLM judgement, given frames and told explicitly that command
   acknowledgements are not evidence.
3. The schema itself. VerificationResult downgrades PASS to INCONCLUSIVE when
   no proof-kind evidence is attached. Even a model determined to be
   optimistic cannot get a green result past it.

Layer 3 is the one that actually holds. Prompts can be ignored; validators
cannot.
"""

from __future__ import annotations

from typing import Any

from base import BaseAgent
from schemas import (
    CriterionResult,
    Evidence,
    EvidenceKind,
    VerificationResult,
    Verdict,
)
from state import AgenticState, note


class VerifierAgent(BaseAgent):
    """Judges the run against its success criteria, using evidence only."""

    role = "verifier"

    def run(self, state: AgenticState) -> dict[str, Any]:
        scenario = state.get("scenario")
        execution = state.get("execution")
        if scenario is None or execution is None:
            raise ValueError("Verifier ran without a scenario or execution.")

        # Layer 1: the deterministic pre-check.
        blocking = self._precheck(scenario, execution, state)
        if blocking is not None:
            return self._emit(blocking, state)

        evidence = self._gather_evidence(execution)
        images = self._select_images(execution)

        prompt = self.render_prompt(
            state,
            scenario=scenario.model_dump(mode="json"),
            execution=execution.model_dump(mode="json"),
            evidence=[e.model_dump(mode="json") for e in evidence],
            criteria=[c.model_dump(mode="json")
                      for c in scenario.success_criteria],
            has_images=bool(images),
            replan_count=int(state.get("replan_count", 0)),
            max_replans=self.context.settings.get("runtime.max_replans", 3),
        )

        # Layer 2.
        result = self.invoke_structured(VerificationResult, prompt, images)
        result.scenario_id = scenario.id

        # Attach the mechanical evidence regardless of what the model returned,
        # so layer 3 judges against the real record rather than the model's
        # account of it.
        result.evidence = _merge_evidence(result.evidence, evidence)

        # Layer 3 runs inside model_validate: re-validating applies the
        # no-pass-without-proof rule to the merged evidence.
        result = VerificationResult.model_validate(
            result.model_dump(mode="python"))

        result.should_replan = self._should_replan(result, state)

        return self._emit(result, state)

    # -- layer 1 -----------------------------------------------------------
    def _precheck(self, scenario: Any, execution: Any,
                  state: AgenticState) -> VerificationResult | None:
        """Cases where the answer is already certain, without an LLM call."""

        if state.get("dry_run"):
            return VerificationResult(
                scenario_id=scenario.id,
                verdict=Verdict.INCONCLUSIVE,
                summary="Dry run: no input was sent and no frames captured.",
                not_proven=["Everything. A dry run validates the plan only."],
            )

        if execution.aborted and not execution.observed_any_change:
            # Nothing ever moved on screen. Whatever the plan intended, the
            # console demonstrably never responded, so this is settled.
            return VerificationResult(
                scenario_id=scenario.id,
                verdict=Verdict.FAIL,
                summary=(
                    "Execution aborted and the screen never changed at any "
                    "point. Nothing reached the console. "
                    f"{execution.abort_reason or ''}").strip(),
                failed_steps=[s.index for s in execution.steps],
                not_proven=[
                    "No success criterion could be evaluated: there is no "
                    "observation of the console responding to anything."],
                evidence=self._gather_evidence(execution),
                confidence=0.95,
            )

        if not execution.steps:
            return VerificationResult(
                scenario_id=scenario.id,
                verdict=Verdict.ERROR,
                summary="No steps were executed at all.",
                not_proven=["The scenario was never exercised."],
            )
        return None

    # -- evidence ----------------------------------------------------------
    @staticmethod
    def _gather_evidence(execution: Any) -> list[Evidence]:
        """Flatten every observation from the run, in order."""
        return [e for step in execution.steps for e in step.evidence]

    def _select_images(self, execution: Any) -> list[dict[str, str]]:
        """Pick a few frames to show a vision model.

        Sending all of them would be slow and expensive, so we take the first
        frame (the starting state), the last (the end state) and the frame from
        the step with the largest change (where the interesting thing happened).
        """
        if not self.context.llm_factory.supports_vision(
                self.spec.get("provider")):
            return []

        paths: list[str] = []
        with_frames = [s for s in execution.steps if s.frame_after]
        if not with_frames:
            return []

        if with_frames[0].frame_before:
            paths.append(with_frames[0].frame_before)

        biggest = max(with_frames, key=lambda s: s.screen_delta or 0.0)
        if biggest.frame_after:
            paths.append(biggest.frame_after)

        if with_frames[-1].frame_after:
            paths.append(with_frames[-1].frame_after)

        images: list[dict[str, str]] = []
        for path in dict.fromkeys(paths):                 # de-dup, keep order
            encoded = self.call_tool("encode_frame_for_vision", frame_path=path)
            if encoded.get("ok"):
                images.append({
                    "base64": encoded["base64"],
                    "media_type": encoded.get("media_type", "image/png"),
                })
        return images

    # -- replanning --------------------------------------------------------
    def _should_replan(self, result: VerificationResult,
                       state: AgenticState) -> bool:
        """Is another attempt worth making?

        Only for failures that look like OUR mistake - a wrong menu route, a
        timing that was too tight. If the rig is broken or the console is
        unresponsive, replanning just repeats the same failure more slowly.
        """
        if result.verdict == Verdict.PASS:
            return False
        if int(state.get("replan_count", 0)) >= int(
                self.context.settings.get("runtime.max_replans", 3)):
            return False

        execution = state.get("execution")
        if execution is not None and not execution.observed_any_change:
            return False              # nothing is getting through; retrying won't help
        return bool(result.should_replan)

    # -- output ------------------------------------------------------------
    def _emit(self, result: VerificationResult,
              state: AgenticState) -> dict[str, Any]:
        self.context.artifacts.save_json(
            "verification.json", result.model_dump(mode="json"))

        return {
            "verification": result,
            "messages": [note(
                self.role,
                f"{result.verdict.value.upper()}: {result.summary}",
                level="info" if result.passed else "error")],
            "agent_outputs": {self.role: {
                "ok": True,
                "verdict": result.verdict.value,
                "passed": result.passed,
                "should_replan": result.should_replan,
            }},
        }


def _merge_evidence(from_model: list[Evidence],
                    measured: list[Evidence]) -> list[Evidence]:
    """Combine model-cited and mechanically-collected evidence.

    Measured evidence goes first: it is the trustworthy part. Model-supplied
    entries are kept because they can add useful reading of a frame, but they
    never replace the record.
    """
    seen = {(e.kind, e.summary) for e in measured}
    extra = [e for e in from_model if (e.kind, e.summary) not in seen]
    return measured + extra
