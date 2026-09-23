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

import json
from typing import Any

from base import BaseAgent
from route_store import invalidate_cached_route, save_cached_route
from schemas import (
    CriterionResult,
    Evidence,
    EvidenceKind,
    StageStatus,
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

        # Layer 1b: Deterministic Compliance Oracle.
        oracle = self._oracle_evaluate(scenario, execution, state)
        if oracle is not None:
            return self._emit(oracle, state)

        evidence = self._gather_evidence(execution)
        images = self._select_images(execution)

        prompt = self.render_prompt(
            state,
            scenario=scenario.model_dump(mode="json"),
            execution=execution.model_dump(mode="json"),
            evidence=[e.model_dump(mode="json") for e in evidence],
            criteria=[c.model_dump(mode="json")
                      for c in scenario.success_criteria],
            stage_summary=[
                item.model_dump(mode="json")
                for item in getattr(execution, "stage_summary", [])
            ],
            last_proven_stage=(
                execution.last_proven_stage.value
                if getattr(execution, "last_proven_stage", None) else None
            ),
            has_images=bool(images),
            replan_count=int(state.get("replan_count", 0)),
            max_replans=self.context.settings.get("runtime.max_replans", 3),
        )

        # Layer 2.
        raw_result = self.invoke_structured(VerificationResult, prompt, images)
        result = _normalise_verification_result(raw_result)
        result.scenario_id = scenario.id

        # Attach the mechanical evidence regardless of what the model returned,
        # so layer 3 judges against the real record rather than the model's
        # account of it.
        result.evidence = _merge_evidence(result.evidence, evidence)

        # Layer 3 runs inside model_validate: re-validating applies the
        # no-pass-without-proof rule to the merged evidence.
        result = VerificationResult.model_validate(
            result.model_dump(mode="python"))
        result.stage_status = self._stage_status_map(execution)
        result.last_proven_stage = getattr(execution, "last_proven_stage", None)
        if result.replan_hint is None:
            result.replan_hint = self._default_replan_hint(execution, result)

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

    def _oracle_evaluate(self, scenario: Any, execution: Any,
                         state: AgenticState) -> VerificationResult | None:
        """Deterministic Compliance Oracle: evaluate success criteria directly.

        Only resolves criteria with a mechanical check_type. Any criterion of
        a different kind is refused, not guessed at.

        Decidable kinds: text_present, no_error_dialog, screen_change.
        Refusing even one required criterion sends the whole scenario to
        layer 2 - a partial deterministic PASS is not produced.
        """
        if getattr(execution, "aborted", False):
            return None
        if not getattr(execution, "observed_any_change", False):
            return None
        if not getattr(execution, "steps", None):
            return None

        # If any step failed, oracle cannot pass
        if any(not getattr(s, "success", True) for s in execution.steps):
            return None

        # If any stage failed or was blocked
        stage_summary = getattr(execution, "stage_summary", []) or []
        if any(item.status in {StageStatus.FAILED, StageStatus.BLOCKED} for item in stage_summary):
            return None

        evidence = self._gather_evidence(execution)
        proofs = [e for e in evidence if e.is_proof]
        if not proofs:
            return None

        required = [c for c in scenario.success_criteria if c.required]
        if not required:
            return None

        # Kinds the oracle is willing to answer without a vision model.
        decidable = {"text_present", "no_error_dialog", "screen_change"}
        if any(self._criterion_kind(c) not in decidable for c in required):
            return None

        joined_text = self._joined_observed_text(proofs)
        results: list[CriterionResult] = []

        for criterion in required:
            kind = self._criterion_kind(criterion)
            params = criterion.parameters or {}

            if kind == "no_error_dialog":
                met, matching = self._check_no_error_dialog(joined_text, proofs)
            elif kind == "text_present":
                expected = str(params.get("text", "")).strip()
                if not expected:
                    return None
                met, matching = self._check_text_present(expected, joined_text, proofs)
            else:
                min_delta = float(params.get("min_delta", 0.0) or 0.0)
                met, matching = self._check_screen_change(execution, min_delta, proofs)

            if not met:
                return None

            results.append(CriterionResult(
                criterion=criterion.description,
                met=True,
                reasoning=(f"Deterministically verified by the compliance oracle "
                          f"(check_type={kind}), no vision model call needed."),
                evidence=matching,
                confidence=0.95,
            ))
        return VerificationResult(
            scenario_id=scenario.id,
            verdict=Verdict.PASS,
            confidence=0.95,
            summary=("Deterministic Compliance Oracle: all required criteria have "
                     "a mechanical check_type and every one was proven by OCR or "
                     "frame-diff evidence. No vision model call was needed."),
            criteria=results,
            not_proven=[],
            evidence=evidence,
            stage_status=self._stage_status_map(execution),
            last_proven_stage=getattr(execution, "last_proven_stage", None),
            should_replan=False,
        )

    @staticmethod
    def _criterion_kind(criterion: Any) -> str:
        return str(getattr(criterion, "check_type", "")
                   or getattr(criterion, "kind", "") or "").strip().lower()

    @staticmethod
    def _joined_observed_text(proofs: list[Evidence]) -> str:
        """Flatten actual observed OCR and vision text into one lowercase corpus."""
        parts: list[str] = []
        for e in proofs:
            if e.kind == EvidenceKind.OCR_TEXT:
                if isinstance(e.detail, dict) and e.detail.get("text"):
                    parts.append(str(e.detail["text"]).lower())
                else:
                    parts.append(str(e.summary).lower())
            elif e.kind == EvidenceKind.VISION_MODEL:
                if isinstance(e.detail, dict) and e.detail.get("observed_text"):
                    parts.append(str(e.detail["observed_text"]).lower())
        return " ".join(parts)

    _ERROR_PHRASES = (
        "something went wrong", "failed to launch", "corrupt file",
        "network error", "connection lost", "an error occurred",
        "could not connect", "please try again later",
    )

    @classmethod
    def _check_no_error_dialog(cls, joined_text: str,
                               proofs: list[Evidence]) -> tuple[bool, list[Evidence]]:
        hit = next((p for p in cls._ERROR_PHRASES if p in joined_text), None)
        if hit is not None:
            return False, []
        matches = [e for e in proofs
                  if e.kind in (EvidenceKind.OCR_TEXT, EvidenceKind.VISION_MODEL)]
        return True, matches or proofs[:1]

    @staticmethod
    def _check_text_present(expected: str, joined_text: str,
                            proofs: list[Evidence]) -> tuple[bool, list[Evidence]]:
        expected_lower = expected.lower().strip()
        if expected_lower not in joined_text:
            return False, []
        matching = [
            e for e in proofs
            if e.kind == EvidenceKind.OCR_TEXT
            and expected_lower in str((e.detail or {}).get("text", e.summary)).lower()
        ]
        if not matching:
            return False, []
        return True, matching

    @staticmethod
    def _check_screen_change(execution: Any, min_delta: float,
                             proofs: list[Evidence]) -> tuple[bool, list[Evidence]]:
        diffs = [e for e in proofs if e.kind == EvidenceKind.SCREEN_DIFF]
        if not diffs:
            return False, []
        if min_delta > 0.0:
            qualifying = [
                e for e in diffs
                if isinstance(e.detail, dict)
                and float(e.detail.get("delta", 0.0) or 0.0) >= min_delta
            ]
            if not qualifying:
                return False, []
            return True, qualifying
        return True, diffs

    @staticmethod
    def _stage_status_map(execution: Any) -> dict[str, StageStatus]:
        return {
            item.stage.value: item.status
            for item in (getattr(execution, "stage_summary", []) or [])
        }

    @staticmethod
    def _default_replan_hint(execution: Any,
                             result: VerificationResult) -> str | None:
        if result.verdict == Verdict.PASS:
            return None
        failed = next(
            (item for item in (getattr(execution, "stage_summary", []) or [])
             if item.status in {StageStatus.FAILED, StageStatus.BLOCKED}),
            None,
        )
        if failed is not None:
            return f"{failed.stage.value} failed: {failed.summary}"
        if getattr(execution, "current_stage", None) is not None:
            return (f"{execution.current_stage.value} was in progress but not "
                    f"proven.")
        return result.summary

    # -- evidence ----------------------------------------------------------
    @staticmethod
    def _gather_evidence(execution: Any) -> list[Evidence]:
        """Flatten every observation from the run, in order."""
        return [e for step in execution.steps for e in step.evidence]

    def _select_images(self, execution: Any) -> list[dict[str, str]]:
        """Pick representative frames across executed stages to show a vision model."""
        if not self.context.llm_factory.supports_vision(
                self.spec.get("provider")):
            return []

        paths: list[str] = []
        with_frames = [s for s in execution.steps if s.frame_after]
        if not with_frames:
            return []

        if with_frames[0].frame_before:
            paths.append(with_frames[0].frame_before)

        # Sample a frame from each distinct stage executed
        stages_seen: set[str] = set()
        for s in with_frames:
            stage_name = str(getattr(s, "stage", "") or "")
            if stage_name and stage_name not in stages_seen:
                stages_seen.add(stage_name)
                if s.frame_after:
                    paths.append(s.frame_after)

        biggest = max(with_frames, key=lambda s: s.screen_delta or 0.0)
        if biggest.frame_after:
            paths.append(biggest.frame_after)

        if with_frames[-1].frame_after:
            paths.append(with_frames[-1].frame_after)

        images: list[dict[str, str]] = []
        for path in list(dict.fromkeys(paths))[:8]:                 # de-dup, keep order, max 8
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

        scenario = state.get("scenario")
        plan = state.get("plan")
        if scenario is not None and plan is not None:
            if result.verdict == Verdict.PASS:
                save_cached_route(self.context.artifacts.run_dir, scenario.id, plan)
            elif result.verdict == Verdict.FAIL:
                invalidate_cached_route(self.context.artifacts.run_dir, scenario.id)

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


def _normalise_verification_result(raw: VerificationResult) -> VerificationResult:
    """Coerce common structured-output near-misses before validation.

    Some providers occasionally return JSON arrays as quoted strings for
    `criteria` / `evidence`. That is a framework-formatting defect, not a test
    verdict, so we normalise it here instead of letting the whole verifier die.
    """
    data = raw.model_dump(mode="python")
    data["criteria"] = _coerce_json_list(data.get("criteria"), "criteria")
    data["evidence"] = _coerce_json_list(data.get("evidence"), "evidence")

    for item in data["criteria"]:
        if isinstance(item, dict):
            item["evidence"] = _coerce_json_list(item.get("evidence"), "criterion evidence")

    return VerificationResult.model_validate(data)


def _coerce_json_list(value: Any, field_name: str) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Verifier returned invalid {field_name}: expected a list or "
                f"JSON array string, got {text[:120]!r}") from exc
        if isinstance(parsed, list):
            return parsed
    raise ValueError(
        f"Verifier returned invalid {field_name}: expected list, got "
        f"{type(value).__name__}.")
