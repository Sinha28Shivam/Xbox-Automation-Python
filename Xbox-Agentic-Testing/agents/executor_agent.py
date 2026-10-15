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
If several consecutive INPUT steps produce zero meaningful evidence of change,
execution stops. That pattern almost always means input is not reaching the
console (an unauthenticated GIMX session being the classic cause), and
hammering forty more buttons at a console that is not listening wastes minutes
and muddies the RCA timeline. Observational steps like capture/wait must not be
counted toward that streak.
"""

from __future__ import annotations

from core.logging_setup import log_suppressed as _log_suppressed

import re
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

from agents.base import BaseAgent
from core.schemas import (
    Evidence,
    EvidenceKind,
    ExecutionResult,
    PlannedStep,
    ScenarioStage,
    StageResult,
    StageStatus,
    StageTransition,
    StepResult,
)
from core import runtime_lessons, text_match
from core.state import AgenticState, note


_INGAME_STAGES = frozenset({
    "menu_detection",
    "level_navigation",
    "level_launch",
    "closed_loop_play",
    "pause_checkpoint",
    "main_menu_return",
    "level_select_replay",
    "achievements_review",
})


_SIGNATURES_PATH = Path(__file__).resolve().parents[1] / "config" / "screen_signatures.yaml"
_SIGNATURES_FALLBACK = {
    "store": ["buy $", "join game pass"],
    "account_prompt": ["your account needs attention"],
    "dashboard": ["my games & apps", "add to play later", "sponsored"],
    "in_game": ["press a to start", "select level"],
}


@lru_cache(maxsize=1)
def _screen_signatures() -> dict[str, list[str]]:
    """Screen-identifying text from config/screen_signatures.yaml.

    Games list their own signatures there, so adding a game is a YAML edit.
    A missing or broken file falls back to a minimal built-in set rather than
    disabling the guard.
    """
    try:
        import yaml
        data = yaml.safe_load(_SIGNATURES_PATH.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        print(f"  [guard] screen_signatures.yaml unreadable ({exc}); using built-ins",
              flush=True)
        return _SIGNATURES_FALLBACK
    def lower(items: Any) -> list[str]:
        return [str(i).lower() for i in (items or [])]
    in_game = lower(data.get("in_game_generic"))
    for phrases in (data.get("games") or {}).values():
        in_game += lower(phrases)
    policy = str(data.get("account_prompt_policy", "warn")).lower()
    return {
        "account_prompt_policy": [policy if policy in ("warn", "block") else "warn"],
        "store": lower(data.get("store")) or _SIGNATURES_FALLBACK["store"],
        "account_prompt": lower(data.get("account_prompt")),
        "dashboard": lower(data.get("dashboard")) or _SIGNATURES_FALLBACK["dashboard"],
        "in_game": in_game or _SIGNATURES_FALLBACK["in_game"],
    }


# Steps that only LOOK at the screen. They never change console state, so the
# before/after/diff/OCR wrapper used for inputs adds time but no evidence.
_OBSERVE_ONLY_ACTIONS = frozenset({
    "capture_frame",
    "wait_for_stable_screen",
    "read_screen_text",
    "find_text_on_screen",
    "check_for_text",
    "read_text_region",
    "detect_focus_highlight",
})


class ExecutorAgent(BaseAgent):
    """Runs the plan against real hardware, capturing evidence throughout."""

    role = "executor"
    uses_llm = False

    @staticmethod
    def _is_ingame_stage(step: PlannedStep) -> bool:
        if step.stage is None:
            return False
        val = step.stage.value if hasattr(step.stage, "value") else str(step.stage)
        return val in _INGAME_STAGES

    def _load_applied_lessons(self, revision: int) -> list[dict[str, Any]]:
        """The exact lessons the planner used for THIS plan revision."""
        path = self.context.artifacts.reports_dir / f"lessons-applied-r{revision}.json"
        if not path.is_file():
            return []
        try:
            import json
            return list(json.loads(path.read_text(encoding="utf-8")).get("lessons") or [])
        except Exception as _exc:
            _log_suppressed("agents/executor_agent.py:137", _exc)
            return []

    def _runtime_lesson_check(self, step: PlannedStep, rules: dict[str, list[str]],
                              lessons: list[dict[str, Any]]) -> dict[str, Any] | None:
        """Read the LIVE screen and let lesson rules veto or annotate the step."""
        args = dict(step.arguments or {})
        if not runtime_lessons.needs_screen(step.action, args, rules, lessons):
            return None
        frame = self.call_tool("capture_frame", label=f"{step.index}-lesson-check")
        frame_path = frame.get("frame_path") if frame.get("ok") else None
        reading = (self.call_tool("read_screen_text", frame_path=frame_path)
                   if frame_path else {"ok": False})
        if not reading.get("ok"):
            # Cannot see the screen: never skip blindly, and say so.
            return {"rule": "unverified", "lessons": [l.get("id") for l in lessons][:3],
                    "action": "proceed", "frame_path": frame_path,
                    "reason": "Screen could not be read; lesson could not be checked."}
        text = str(reading.get("text", ""))
        decision = runtime_lessons.decide(step.action, args, text, rules, lessons,
                                          destinations=getattr(self, "_destinations", []))
        if decision is not None:
            decision["frame_path"] = frame_path
            decision["screen_text"] = text[:800]
            decision["screen_preview"] = " | ".join(
                l.strip() for l in text.splitlines() if l.strip())[:160]
        return decision

    @staticmethod
    def _dashboard_is_expected(scenario: Any) -> bool:
        tags = {str(t).lower() for t in (getattr(scenario, "tags", None) or [])}
        return bool(tags & {"dashboard", "guide"})

    @staticmethod
    def _guard_ingame_screen(prior_results: list[StepResult]) -> dict[str, Any]:
        """Verify console is in an active game state before dispatching in-game actions.

        If the console was redirected to the Microsoft Store / Game Hub, an account error
        dialog, or remained stuck on the Xbox Dashboard, dispatching gameplay buttons
        (like navigating or pressing A) would be blind tapping. This guard halts execution
        immediately.
        """
        if not prior_results:
            return {"ok": True}

        last_with_ocr = next((r for r in reversed(prior_results) if r.ocr_text), None)
        if last_with_ocr is None:
            return {"ok": True}

        text = (last_with_ocr.ocr_text or "").lower()
        sigs = _screen_signatures()

        # 1. Store / Purchase screen (config/screen_signatures.yaml: store).
        # Exact match: prices and "buy $" must not be guessed at.
        hit_store = next((p for p in sigs["store"] if p in text), None)
        if hit_store:
            return {
                "ok": False,
                "error": f"Blind tapping prevented: Console is displaying Microsoft Store / Game Hub ('{hit_store}') instead of the running game. Account may lack active entitlement.",
                "observation": "Refused to dispatch gameplay controls on Microsoft Store screen.",
            }

        # 2. Account attention / ownership prompt. Policy is explicit in
        # config/screen_signatures.yaml (account_prompt_policy): "warn" keeps
        # going (the long-standing operator override), "block" stops.
        hit_error = text_match.first_match(text, sigs["account_prompt"])
        if hit_error:
            if sigs.get("account_prompt_policy") == ["block"]:
                return {
                    "ok": False,
                    "error": f"Blind tapping prevented: account prompt on screen ('{hit_error}').",
                    "observation": "Refused to dispatch gameplay controls on an account prompt.",
                }
            print(f"  [guard] Account attention prompt ('{hit_error}') detected - continuing "
                  f"(account_prompt_policy: warn).", flush=True)

        # 3. Dashboard screen (without game running)
        hit_dash = text_match.first_match(text, sigs["dashboard"])
        # Fuzzy on purpose: a game signature misread by OCR ("Minecraft
        # Dundecns") must still count, or the guard aborts a running game.
        has_game_sig = text_match.first_match(text, sigs["in_game"]) is not None
        if hit_dash and not has_game_sig:
            return {
                "ok": False,
                "error": f"Blind tapping prevented: Console is on the Xbox Dashboard Home ('{hit_dash}') instead of the running game.",
                "observation": "Refused to dispatch gameplay controls on dashboard home screen.",
            }

        return {"ok": True}

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
        focus_unproven = False
        stage_state: dict[ScenarioStage, StageResult] = {}
        transitions: list[StageTransition] = []
        current_stage: ScenarioStage | None = None
        last_proven_stage: ScenarioStage | None = None

        revision = int(getattr(plan, "revision", 1) or 1)
        lessons = self._load_applied_lessons(revision)
        rules = runtime_lessons.active_rules(lessons)
        # Leaf destinations only: items that are OPENED (require_focus). The
        # Game Activity tab heads every screen its items are opened from.
        self._destinations = [
            str(nav["target_text"]) for c in scenario.success_criteria
            for nav in [(c.parameters or {}).get("navigation_guidance")]
            if isinstance(nav, dict) and nav.get("target_text")
            and nav.get("require_focus", True)]
        runtime_log: list[dict[str, Any]] = []
        if lessons:
            print(f"  [executor] {len(lessons)} lesson(s) active at runtime"
                  + (f"; rules: {', '.join(rules)}" if rules else ""), flush=True)

        for step in plan.steps[:max_steps]:
            if time.time() > run_deadline:
                aborted, abort_reason = True, (
                    f"Run timeout exceeded after {len(results)} steps.")
                print(f"\n{'!' * 72}", flush=True)
                print(f"[TIMEOUT EXCEEDED at Step {len(results)}/{len(plan.steps)}]", flush=True)
                print(f"Reason: {abort_reason}", flush=True)
                print(f"{'!' * 72}\n", flush=True)
                break

            # Rich real-time terminal log header
            stage_name = step.stage.value if step.stage is not None else "unspecified"
            print("\n" + "-" * 72, flush=True)
            print(f"[Step {step.index + 1}/{len(plan.steps)}] [{stage_name.upper()}] Action: {step.action}", flush=True)
            if step.arguments:
                print(f"  Arguments: {step.arguments}", flush=True)
            if step.intent:
                print(f"  Intent:    {step.intent}", flush=True)
            if step.expected_observation:
                print(f"  Expected:  {step.expected_observation}", flush=True)

            # In-game active screen guard: prevent blind tapping on dashboard or store.
            # Skipped for dashboard/Guide scenarios: there the dashboard IS the
            # product under test, not a sign the game was lost.
            # (run-20261015-113636 aborted on the exact dashboard its last
            # criterion asked to verify.)
            if not dry_run and self._is_ingame_stage(step) \
                    and not self._dashboard_is_expected(state.get("scenario")):
                screen_guard = self._guard_ingame_screen(results)
                if not screen_guard["ok"]:
                    blocked = StepResult(
                        index=step.index,
                        action=step.action,
                        arguments=dict(step.arguments),
                        stage=step.stage,
                        dispatched=False,
                        error=screen_guard["error"],
                        observation=screen_guard["observation"],
                    )
                    results.append(blocked)
                    aborted = True
                    abort_reason = blocked.error or "blocked"
                    notes.append(abort_reason)
                    print(f"\n{'!' * 72}", flush=True)
                    print(f"[BLIND TAPPING PREVENTED at Step {step.index + 1}]", flush=True)
                    print(f"Reason: {abort_reason}", flush=True)
                    print(f"{'!' * 72}\n", flush=True)
                    break

            if focus_unproven and self._is_confirm_after_focus_move(step):
                blocked = StepResult(
                    index=step.index,
                    action=step.action,
                    arguments=dict(step.arguments),
                    dispatched=False,
                    error=("Selection blocked: the previous focus-navigation step "
                           "did not prove which tile/menu item was selected."),
                    observation=("Refused to press confirm because focus on the "
                                 "intended target was not visually proven."),
                )
                results.append(blocked)
                aborted = True
                abort_reason = blocked.error or "blocked"
                notes.append(abort_reason)
                print(f"\n{'!' * 72}", flush=True)
                print(f"[SELECTION BLOCKED at Step {step.index + 1}]", flush=True)
                print(f"Reason: {abort_reason}", flush=True)
                print(f"{'!' * 72}\n", flush=True)
                break

            decision = (None if dry_run or not lessons else
                        self._runtime_lesson_check(step, rules, lessons))
            if decision is not None:
                decision["step_index"] = step.index
                runtime_log.append(decision)
                print(f"  [lesson] {decision['action'].upper()} "
                      f"({decision['rule']}, {', '.join(decision['lessons'])}): "
                      f"{decision['reason']}", flush=True)
            if decision is not None and decision["action"] == "skip":
                result = StepResult(
                    index=step.index, action=step.action,
                    arguments=dict(step.arguments), stage=step.stage,
                    dispatched=False,
                    observation=(f"Skipped by runtime lesson "
                                 f"{', '.join(decision['lessons'])}: {decision['reason']}"),
                    frame_before=decision.get("frame_path"),
                    frame_after=decision.get("frame_path"),
                    ocr_text=decision.get("screen_text"),
                )
                results.append(result)
                notes.append(result.observation)
                print("  Result:    SKIPPED (lesson)", flush=True)
                continue

            result = self._execute_step(step, dry_run)
            results.append(result)
            current_stage, last_proven_stage = self._update_stage_tracking(
                step, result, stage_state, transitions, current_stage, last_proven_stage)
            focus_unproven = (
                self._is_focus_navigation_step(step)
                and bool(result.error)
                and "focus" in (result.error or "").lower()
            ) or self._is_unproven_focus_check(step, result)

            status_label = f"FAILED: {result.error}" if result.error else "OK"
            print(f"  Result:    {status_label} ({result.duration_seconds:.2f}s)", flush=True)
            if result.screen_delta is not None:
                print(f"  Delta:     {result.screen_delta:.3f}", flush=True)
            if result.ocr_text:
                first_lines = [l.strip() for l in result.ocr_text.splitlines() if l.strip()][:3]
                print(f"  OCR Text:  {' | '.join(first_lines)[:120]}", flush=True)
            if result.observation:
                print(f"  Observe:   {result.observation[:120]}", flush=True)

            # Only a HARD failure aborts. A step that merely observed nothing
            # useful is recorded and the run continues.
            if result.error and not step.optional and self._is_fatal(result):
                aborted, abort_reason = True, (
                    f"Step {step.index + 1} ({step.action}) failed: {result.error}")
                print(f"\n{'!' * 72}", flush=True)
                print(f"[FATAL STEP FAILURE at Step {step.index + 1}]", flush=True)
                print(f"Reason: {abort_reason}", flush=True)
                print(f"{'!' * 72}\n", flush=True)
                break

            if result.error:
                notes.append(
                    f"Step {step.index} ({step.action}) reported an error but "
                    f"the run continued: {result.error}")

            # Only actionable input steps count toward the "dead input" abort.
            if self._counts_toward_dead_input(step):
                if self._shows_positive_progress(step, result):
                    unchanged_streak = 0
                else:
                    unchanged_streak += 1
            else:
                unchanged_streak = 0

            if unchanged_streak >= 3 and not dry_run:
                aborted = True
                abort_reason = (
                    f"{unchanged_streak} consecutive input actions produced no "
                    f"meaningful visual progress. Input is likely not "
                    f"reaching the console - the usual cause is a GIMX session "
                    f"that was never authenticated with the Guide button. "
                    f"Stopping rather than sending more input into a void.")
                notes.append(abort_reason)
                print(f"\n{'!' * 72}", flush=True)
                print(f"[DEAD INPUT ABORT at Step {step.index + 1}]", flush=True)
                print(f"Reason: {abort_reason}", flush=True)
                print(f"{'!' * 72}\n", flush=True)
                break

        if aborted:
            print(f"\n{'!' * 72}", flush=True)
            print(f"[EXECUTOR ABORTED at Step {len(results)}/{len(plan.steps)}]", flush=True)
            print(f"Reason: {abort_reason}", flush=True)
            print(f"{'!' * 72}\n", flush=True)
        else:
            print(f"\n[EXECUTOR COMPLETED] {len(results)}/{len(plan.steps)} steps executed in {round(time.time() - started, 2)}s\n", flush=True)

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
            current_stage=current_stage,
            last_proven_stage=last_proven_stage,
            stage_transitions=transitions,
            stage_summary=list(stage_state.values()),
        )

        # execution.json always holds the latest attempt (what the verifier
        # judged); execution-r<N>.json keeps every attempt so a replan can no
        # longer erase the progress an earlier attempt made.
        execution_payload = execution.model_dump(mode="json")
        self.context.artifacts.save_json("execution.json", execution_payload)
        self.context.artifacts.save_json(
            f"execution-r{getattr(plan, 'revision', 1)}.json", execution_payload)
        if lessons:
            self.context.artifacts.save_json(
                f"runtime-lessons-r{revision}.json",
                {"lessons": [l.get("id") for l in lessons], "rules": rules,
                 "decisions": [{k: v for k, v in d.items() if k != "screen_text"}
                               for d in runtime_log]})

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
                            arguments=dict(step.arguments),
                            stage=step.stage,
                            stage_goal=step.stage_goal,
                            progress_signal=step.progress_signal,
                            replan_on=list(step.replan_on),
                            stage_status=(
                                StageStatus.IN_PROGRESS
                                if step.stage is not None else
                                StageStatus.NOT_STARTED
                            ))

        observe_only = step.action in _OBSERVE_ONLY_ACTIONS
        before_path = (None if observe_only else
                       self._capture(f"{step.index}-before-{step.action}",
                                     step.index, dry_run))
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

        if observe_only and not dry_run:
            # The step itself captured/read the screen and changed nothing, so
            # its own frame and text are the evidence. Wrapping it in a second
            # capture + stability wait + diff + OCR (as for inputs) cost ~5.8 s
            # x 21 steps in run-20261015-113636 for no new information.
            self._adopt_own_observation(result, tool_result)
        elif step.verify and not dry_run:
            self._observe(step, result)
            self._read_text(result)
        if result.error:
            result.stage_status = (
                StageStatus.FAILED if step.stage is not None else result.stage_status)
        elif step.stage is not None and self._step_proved_stage_progress(step, result):
            result.stage_status = StageStatus.PROVEN

        result.duration_seconds = round(time.time() - started, 2)
        return result

    def _adopt_own_observation(self, result: StepResult,
                               tool_result: dict[str, Any]) -> None:
        """Use an observe-only step's own frame/text as its evidence."""
        frame = (tool_result.get("frame_path")
                 or self.context.tools.context.scratch.get("last_frame_path"))
        result.frame_after = frame
        result.frame_before = frame
        text = tool_result.get("text") or tool_result.get("screen_text")
        if text:
            result.ocr_text = str(text)
            result.ocr_engine = str(tool_result.get("engine") or "")
            result.evidence.append(Evidence(
                kind=EvidenceKind.OCR_TEXT,
                summary=f"Read {len(str(text).splitlines())} line(s) of on-screen text",
                detail={"text": str(text)[:2000]},
                frame_path=frame, source_tool=result.action))
        elif frame:
            # Let the verifier still see text for this frame (cached OCR).
            self._read_text(result)
        result.observation = (
            f"Observed the screen ({result.action}); no input was sent."
            if not result.error else result.observation)

    @staticmethod
    def _step_proved_stage_progress(step: PlannedStep, result: StepResult) -> bool:
        if step.stage is None or result.error:
            return False
        proof_seen = any(e.is_proof for e in result.evidence)
        if step.action == "detect_focus_highlight" and not result.error:
            return proof_seen
        if step.progress_signal:
            return bool(result.frame_after and proof_seen)
        return bool(result.frame_after and proof_seen)

    @staticmethod
    def _update_stage_tracking(
        step: PlannedStep,
        result: StepResult,
        stage_state: dict[ScenarioStage, StageResult],
        transitions: list[StageTransition],
        current_stage: ScenarioStage | None,
        last_proven_stage: ScenarioStage | None,
    ) -> tuple[ScenarioStage | None, ScenarioStage | None]:
        stage = step.stage
        if stage is None:
            return current_stage, last_proven_stage

        if stage not in stage_state:
            stage_state[stage] = StageResult(
                stage=stage,
                status=StageStatus.NOT_STARTED,
                summary=step.stage_goal or step.intent,
            )

        stage_result = stage_state[stage]
        if current_stage != stage:
            current_stage = stage

        stage_result.status = result.stage_status
        stage_result.summary = result.observation or step.expected_observation
        if result.frame_after:
            stage_result.evidence = list(dict.fromkeys(
                [*stage_result.evidence, result.frame_after]))

        if result.stage_status == StageStatus.PROVEN:
            transitions.append(StageTransition(
                from_stage=last_proven_stage,
                to_stage=stage,
                evidence=[
                    path for path in [result.frame_after]
                    if path
                ],
            ))
            last_proven_stage = stage
        return current_stage, last_proven_stage

    @staticmethod
    def _is_fatal(result: StepResult) -> bool:
        """Should this error stop the whole run?

        Fatal: the hardware layer is gone (capture device lost, controller
        unavailable), or core scenario lifecycle steps failed (game launch failed,
        game not found on dashboard, store/purchase redirect, account error).
        Continuing past a failed launch results in blind input sent to the dashboard/store.
        """
        text = (result.error or "").lower()
        fatal_signs = (
            "capture unavailable",
            "controller unavailable",
            "adapter",
            "device may have been taken",
            "game launch failed",
            "not visually identified",
            "unrecovered_error_dialog",
            "refused to press confirm",
            "selection blocked",
            "blind tapping prevented",
        )
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
        if lines:
            txt_summary = " | ".join(lines[:3])[:100]
            if result.observation:
                result.observation += f" [OCR: {txt_summary}]"
            else:
                result.observation = f"[OCR: {txt_summary}]"

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
            after_path=after_path,
            threshold=self._screen_change_threshold(step))

        if not comparison.get("ok"):
            result.observation = f"Comparison failed: {comparison.get('error')}"
            return

        delta = float(comparison.get("delta", 0.0))
        changed = bool(comparison.get("changed"))
        result.screen_delta = delta
        if self._is_focus_navigation_step(step):
            focus_check = self._verify_focus_target(step, after_path)
            if focus_check is not None:
                result.evidence.append(focus_check["evidence"])
                is_matched = bool(focus_check.get("matched", focus_check.get("ok", False)))
                if is_matched:
                    result.observation = (
                        f"Focus highlight matched target '{focus_check['target']}'. "
                        f"Expected: {step.expected_observation}")
                    result.error = None
                else:
                    result.observation = (
                        f"Focus movement not proven. Expected target "
                        f"'{focus_check['target']}', observed "
                        f"'{focus_check['observed']}'.")
                    result.error = (
                        result.error or
                        "Focus movement could not be proven on the intended target.")
            elif (not changed):
                result.observation = (
                    f"Low-motion focus transition was not proven by whole-screen "
                    f"diff (delta {delta:.3f}). Expected: {step.expected_observation}")
                result.error = (
                    result.error or
                    "Focus movement could not be proven on a low-motion screen.")
            else:
                result.observation = (
                    f"Screen changed (delta {delta:.3f}), but no target label was "
                    f"available to prove focus. Expected: {step.expected_observation}")
                result.error = (
                    result.error or
                    "Focus movement changed the screen but did not prove the selected target.")
        else:
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

    @staticmethod
    def _counts_toward_dead_input(step: PlannedStep) -> bool:
        return step.action in {
            "press_button",
            "hold_button",
            "run_sequence",
            "type_text",
            "move_stick",
            "pull_trigger",
        }

    def _shows_positive_progress(self, step: PlannedStep, result: StepResult) -> bool:
        threshold = self._screen_change_threshold(step)
        if self._is_focus_navigation_step(step) and not result.error:
            return True
        if result.screen_delta is not None and result.screen_delta >= threshold:
            return True
        if step.action == "type_text" and result.dispatched and not result.error:
            return True
        if self._is_low_motion_focus_step(step):
            haystack = " ".join(filter(None, [
                result.observation or "",
                result.ocr_text or "",
                step.expected_observation or "",
            ])).lower()
            progress_markers = (
                "top results",
                "installed",
                "games",
                "people",
                "apps",
                "settings",
                "minecraft",
                "result tile",
            )
            if any(marker in haystack for marker in progress_markers):
                return True
        return False

    @staticmethod
    def _is_low_motion_focus_step(step: PlannedStep) -> bool:
        if step.action != "press_button":
            return False
        button = str(step.arguments.get("button", "")).lower()
        if button not in {"up", "down", "left", "right"}:
            return False
        text = " ".join([
            str(step.intent or ""),
            str(step.expected_observation or ""),
        ]).lower()
        markers = (
            "focus moves",
            "highlight",
            "result tile",
            "results list",
            "search field",
            "first result",
        )
        return any(marker in text for marker in markers)

    def _is_focus_navigation_step(self, step: PlannedStep) -> bool:
        if not self._is_low_motion_focus_step(step):
            return False
        return bool(self._extract_expected_focus_label(step))

    @staticmethod
    def _is_confirm_after_focus_move(step: PlannedStep) -> bool:
        return (
            step.action == "press_button"
            and str(step.arguments.get("button", "")).lower() in {"a", "cross"}
        )

    @staticmethod
    def _is_unproven_focus_check(step: PlannedStep, result: StepResult) -> bool:
        if step.action != "detect_focus_highlight" or not result.error:
            return False

        error = (result.error or "").lower()
        no_target_phrases = (
            "no expected label", "expected_label is empty",
            "no target", "nothing to match", "no label supplied",
            "no green-highlight region was detected",
            "in-game",
        )
        if any(phrase in error for phrase in no_target_phrases):
            return False
        return True

    def _verify_focus_target(self, step: PlannedStep,
                             after_path: str) -> dict[str, Any] | None:
        target = self._extract_expected_focus_label(step)
        if not target:
            return None
        detection = self.call_tool(
            "detect_focus_highlight",
            frame_path=after_path,
            expected_label=target,
        )
        observed = (
            detection.get("selected_label")
            or detection.get("observed_text")
            or ""
        )
        detail = _jsonable(detection)
        detail["expected_label"] = target
        evidence = Evidence(
            kind=EvidenceKind.VISION_MODEL if detection.get("ok") else EvidenceKind.FRAME_STATS,
            summary=(
                f"Focus highlight {'matched' if detection.get('ok') else 'did not match'} "
                f"expected '{target}' (observed '{observed}')."
            ),
            detail=detail,
            frame_path=after_path,
            confidence=0.85 if detection.get("ok") else 0.4,
            source_tool="detect_focus_highlight",
        )
        return {
            "ok": bool(detection.get("ok", False)),
            "matched": bool(detection.get("ok", False) and detection.get("matched", True)),
            "target": target,
            "observed": observed[:200],
            "evidence": evidence,
        }

    @staticmethod
    def _extract_expected_focus_label(step: PlannedStep) -> str:
        for key in ("expected_label", "target", "label", "text"):
            value = str(step.arguments.get(key, "") or "").strip()
            if value and not value.startswith("stage-"):
                return value

        text = " ".join([
            str(step.intent or ""),
            str(step.expected_observation or ""),
        ])
        quoted = re.findall(r"['\"]([^'\"]{2,80})['\"]", text)
        if quoted:
            return quoted[-1].strip()

        match = re.search(
            r"(?:highlight(?:ed)?|focus(?:ed)?|target)\s+(?:is|on|to)?\s*([A-Za-z0-9 &/+-]{3,60})",
            text,
            flags=re.IGNORECASE,
        )
        if match:
            return match.group(1).strip(" .")
        return ""

    def _screen_change_threshold(self, step: PlannedStep) -> float:
        default = float(
            self.context.settings.get("verification.screen_change_threshold", 0.5))
        if self._is_low_motion_focus_step(step):
            return min(default, 0.1)
        return default

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
        except Exception as _exc:
            _log_suppressed("agents/executor_agent.py:936", _exc)
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
