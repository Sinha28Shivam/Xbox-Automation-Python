"""
schemas.py - the contracts between agents.

WHY STRUCTURED OUTPUT
---------------------
Agents hand results to other agents. If the executor returns prose, the verifier
has to re-parse English, and a hallucinated "everything worked!" flows straight
into the report. Pydantic models make each handoff explicit and machine-checked:
the verifier receives `passed: bool` plus `evidence: list[Evidence]`, and a
verdict with no evidence attached is rejected by validation before any human
reads it.

THE ONE RULE ENCODED HERE
-------------------------
From docs/07-lessons-learned: "GIMX accepted the event" must never count as a
pass. That rule is not left to a prompt - it is enforced in `Evidence` (which
records HOW something was observed) and in `VerificationResult.passed`, which a
validator forces to False when no observational evidence exists.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ===========================================================================
# Enumerations
# ===========================================================================
class Verdict(str, Enum):
    """Outcome of a run or a step.

    BLOCKED is first-class and important: when the rig itself is broken, both
    PASS and FAIL are lies about the product under test. Reporting BLOCKED is
    what stops a dead capture card from being logged as a product bug.
    """
    PASS = "pass"
    FAIL = "fail"
    BLOCKED = "blocked"
    INCONCLUSIVE = "inconclusive"
    SKIPPED = "skipped"
    ERROR = "error"


class Severity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class EvidenceKind(str, Enum):
    """How an observation was made.

    COMMAND_ACK is listed so it can be explicitly EXCLUDED from proof. It means
    "GIMX accepted the event", which says nothing about the console. Keeping it
    in the taxonomy (rather than omitting it) means the distinction is visible
    in the report instead of being invisible.
    """
    SCREEN_DIFF = "screen_diff"        # pixels changed after the action
    OCR_TEXT = "ocr_text"              # text read from the frame
    SCREEN_MATCH = "screen_match"      # frame matched a reference screen
    VISION_MODEL = "vision_model"      # a vision LLM described the frame
    FRAME_STATS = "frame_stats"        # brightness/std/tones
    DEVICE_STATE = "device_state"      # probe of GIMX / capture device
    LOG_LINE = "log_line"              # a line from a tool's output
    COMMAND_ACK = "command_ack"        # NOT proof of anything on screen
    TIMING = "timing"


# Only these kinds may support a PASS verdict.
PROOF_KINDS: frozenset[EvidenceKind] = frozenset({
    EvidenceKind.SCREEN_DIFF,
    EvidenceKind.OCR_TEXT,
    EvidenceKind.SCREEN_MATCH,
    EvidenceKind.VISION_MODEL,
})


class FailureClass(str, Enum):
    """Root cause categories.

    The distinction that matters most in practice is RIG_FAULT vs PRODUCT_DEFECT.
    Confusing the two either raises bugs against a healthy console or hides real
    ones behind "the test rig is flaky".
    """
    RIG_FAULT = "rig_fault"                    # GIMX, adapter, capture, cabling
    AUTOMATION_DEFECT = "automation_defect"    # our plan/timing/selector is wrong
    PRODUCT_DEFECT = "product_defect"          # the console genuinely misbehaved
    ENVIRONMENT = "environment"                # network, updates, HDCP, account
    FLAKY_TIMING = "flaky_timing"              # real but intermittent
    SCENARIO_DEFECT = "scenario_defect"        # the test itself is wrong
    UNKNOWN = "unknown"


# ===========================================================================
# Evidence
# ===========================================================================
class Evidence(BaseModel):
    """One observation. The atom every verdict is built from."""

    kind: EvidenceKind
    summary: str = Field(description="One line a human can read")
    detail: dict[str, Any] = Field(default_factory=dict)
    frame_path: str | None = Field(
        default=None, description="Screenshot backing this observation")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    timestamp: str = Field(default_factory=_utc_now)
    source_tool: str | None = None

    @property
    def is_proof(self) -> bool:
        """True if this observation may support a PASS."""
        return self.kind in PROOF_KINDS


# ===========================================================================
# 1. Health
# ===========================================================================
class ComponentHealth(BaseModel):
    name: str
    ok: bool
    detail: str = ""
    metrics: dict[str, Any] = Field(default_factory=dict)
    remediation: str | None = Field(
        default=None, description="What a human should do about it")


class HealthReport(BaseModel):
    """Whether the rig can be trusted right now."""

    healthy: bool
    components: list[ComponentHealth] = Field(default_factory=list)
    blocking_issues: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    recoverable: bool = Field(
        default=False,
        description="Might a bounded automated retry fix this?")
    # Reachability is not authentication. GIMX answers UDP happily while the
    # Guide handshake was never done, and every event then reports 'ok' and
    # goes nowhere. We record the two facts separately and never conflate them.
    gimx_reachable: bool = False
    gimx_authenticated: bool | None = Field(
        default=None,
        description="None = unknown. Only a visible screen change proves it.")
    capture_has_signal: bool = False
    summary: str = ""
    evidence: list[Evidence] = Field(default_factory=list)
    checked_at: str = Field(default_factory=_utc_now)


# ===========================================================================
# 2. Scenario
# ===========================================================================
class SuccessCriterion(BaseModel):
    """A pass condition that must be *observable*."""

    description: str
    check_type: str = Field(
        description="screen_change | text_present | text_absent | "
                    "screen_match | vision_judgement | no_error_dialog")
    parameters: dict[str, Any] = Field(default_factory=dict)
    required: bool = True

    @field_validator("description")
    @classmethod
    def _must_be_observable(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("A success criterion cannot be empty")
        return v


class ValidatedScenario(BaseModel):
    """A scenario normalised into something executable."""

    id: str
    title: str
    description: str = ""
    console: str | None = Field(
        default=None, description="Profile from controls.yaml; None = default")
    goal: str = Field(description="Plain-language objective for the executor")
    preconditions: list[str] = Field(default_factory=list)
    success_criteria: list[SuccessCriterion] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    timeout_seconds: float | None = None
    max_steps: int | None = None

    valid: bool = True
    issues: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    normalised_from: str = Field(
        default="", description="yaml | natural_language | requirement_yaml")
    unresolved_controls: list[str] = Field(
        default_factory=list,
        description="Names not found in controls.yaml - a scenario defect")

    @model_validator(mode="after")
    def _needs_a_criterion(self) -> "ValidatedScenario":
        # A scenario with no observable success criterion can never fail, and a
        # test that cannot fail is worthless. Reject it at the door.
        if self.valid and not self.success_criteria:
            self.valid = False
            self.issues.append(
                "No success criteria: this test could never fail, so it "
                "cannot pass either.")
        return self


# ===========================================================================
# 2b. Requirement input
# ===========================================================================
class RequirementItem(BaseModel):
    """Minimal YAML requirement input for agentic normalization."""

    id: str
    title: str
    goal: str
    expected_outcome: str
    preconditions: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    console: str | None = None
    priority: str | int | None = None

    @field_validator("id", "title", "goal", "expected_outcome")
    @classmethod
    def _required_text(cls, v: str) -> str:
        if not str(v).strip():
            raise ValueError("This field is required and cannot be empty")
        return str(v).strip()


# ===========================================================================
# 3. Plan
# ===========================================================================
class PlannedStep(BaseModel):
    """One action plus the observation that proves it worked."""

    index: int
    action: str = Field(
        description="Tool to call, e.g. press_button / wait_for_text / capture_frame")
    arguments: dict[str, Any] = Field(default_factory=dict)
    intent: str = Field(description="Why this step exists")
    expected_observation: str = Field(
        description="What should be visible afterwards. Required - a step with "
                    "no expected observation cannot be verified.")
    verify: bool = True
    optional: bool = Field(
        default=False, description="Failure here does not fail the run")
    timeout_seconds: float | None = None
    retry_limit: int = 1

    @field_validator("expected_observation")
    @classmethod
    def _no_blind_steps(cls, v: str) -> str:
        if not v.strip():
            raise ValueError(
                "expected_observation is required: an unverifiable step is how "
                "false passes get in.")
        return v


class TestPlan(BaseModel):
    scenario_id: str
    steps: list[PlannedStep] = Field(default_factory=list)
    rationale: str = ""
    assumptions: list[str] = Field(
        default_factory=list,
        description="Stated so the RCA agent can question them later")
    estimated_duration_seconds: float | None = None
    revision: int = Field(default=1, description="Bumped on each replan")
    replan_reason: str | None = None


# ===========================================================================
# 4. Execution
# ===========================================================================
class StepResult(BaseModel):
    index: int
    action: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    # "The command was accepted" - deliberately NOT the same field as `verified`.
    dispatched: bool = False
    error: str | None = None
    observation: str = Field(
        default="", description="What was actually seen afterwards")
    evidence: list[Evidence] = Field(default_factory=list)
    frame_before: str | None = None
    frame_after: str | None = None
    screen_delta: float | None = Field(
        default=None, description="Mean absolute pixel difference")
    ocr_text: str | None = Field(
        default=None,
        description="Text read from the 'after' frame. Captured for every "
                    "step so the verifier can check text criteria and the "
                    "report can show what was actually on screen.")
    ocr_engine: str | None = None
    duration_seconds: float = 0.0
    attempts: int = 1
    started_at: str = Field(default_factory=_utc_now)


class ExecutionResult(BaseModel):
    """What the executor did. Explicitly NOT a verdict."""

    scenario_id: str
    completed: bool = Field(
        default=False, description="Every planned step was attempted")
    aborted: bool = False
    abort_reason: str | None = None
    steps: list[StepResult] = Field(default_factory=list)
    total_steps: int = 0
    dispatched_steps: int = 0
    duration_seconds: float = 0.0
    action_log: list[dict[str, Any]] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    @property
    def observed_any_change(self) -> bool:
        """Did the screen EVER change? If not, nothing reached the console."""
        return any(
            (s.screen_delta or 0) > 0 or
            any(e.is_proof for e in s.evidence)
            for s in self.steps)


# ===========================================================================
# 5. Verification
# ===========================================================================
class CriterionResult(BaseModel):
    criterion: str
    met: bool
    reasoning: str = ""
    evidence: list[Evidence] = Field(default_factory=list)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class VerificationResult(BaseModel):
    """The verdict, with the evidence rule enforced in code."""

    scenario_id: str
    verdict: Verdict
    passed: bool = False
    criteria: list[CriterionResult] = Field(default_factory=list)
    failed_steps: list[int] = Field(default_factory=list)
    summary: str = ""
    # Stating what was NOT proven is as valuable as the verdict. A pass that
    # only checked "something changed" is much weaker than one that read the
    # expected text, and the report should say so.
    not_proven: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    should_replan: bool = False
    replan_hint: str | None = None

    @model_validator(mode="after")
    def _no_pass_without_proof(self) -> "VerificationResult":
        """A PASS requires at least one observational proof.

        This is the single most important invariant in the framework. It is
        enforced here, in code, rather than being asked for in a prompt -
        because a model that decides to be optimistic must not be able to
        produce a green run on its own say-so.
        """
        all_ev = list(self.evidence) + [
            e for c in self.criteria for e in c.evidence]
        if self.verdict == Verdict.PASS and not any(e.is_proof for e in all_ev):
            self.verdict = Verdict.INCONCLUSIVE
            self.passed = False
            self.not_proven.append(
                "Downgraded PASS -> INCONCLUSIVE: no observational evidence "
                "(screen diff, OCR, screen match or vision) was supplied. "
                "Command acknowledgements alone never prove console behaviour.")
        else:
            self.passed = self.verdict == Verdict.PASS
        return self


# ===========================================================================
# 6. Root cause analysis
# ===========================================================================
class Hypothesis(BaseModel):
    statement: str
    failure_class: FailureClass
    likelihood: float = Field(ge=0.0, le=1.0)
    supporting_evidence: list[str] = Field(default_factory=list)
    contradicting_evidence: list[str] = Field(default_factory=list)
    # A hypothesis you cannot check is just an opinion.
    next_check: str = Field(
        description="A concrete action that would confirm or eliminate this")


class RootCauseAnalysis(BaseModel):
    scenario_id: str
    primary_cause: str
    failure_class: FailureClass = FailureClass.UNKNOWN
    severity: Severity = Severity.MEDIUM
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    ruled_out: list[str] = Field(default_factory=list)
    timeline: list[str] = Field(
        default_factory=list, description="Ordered account of what happened")
    recommendations: list[str] = Field(default_factory=list)
    is_retryable: bool = False
    retry_hint: str | None = None
    product_bug_suspected: bool = Field(
        default=False,
        description="Only true when the rig was healthy AND the console "
                    "demonstrably misbehaved")


# ===========================================================================
# 7. Report
# ===========================================================================
class TestReport(BaseModel):
    run_id: str
    scenario_id: str
    scenario_title: str = ""
    requirement_id: str | None = None
    requirement_title: str = ""
    requirement_goal: str = ""
    verdict: Verdict
    summary: str = ""
    started_at: str = ""
    finished_at: str = Field(default_factory=_utc_now)
    duration_seconds: float = 0.0

    health: HealthReport | None = None
    plan: TestPlan | None = None
    execution: ExecutionResult | None = None
    verification: VerificationResult | None = None
    rca: RootCauseAnalysis | None = None

    artifacts: list[str] = Field(default_factory=list)
    screenshots: list[str] = Field(default_factory=list)
    report_files: dict[str, str] = Field(default_factory=dict)
    caveats: list[str] = Field(
        default_factory=list,
        description="Limits of this result - what it does NOT prove")
    metrics: dict[str, Any] = Field(default_factory=dict)


# ===========================================================================
# 8. Control-flow schemas
# ===========================================================================
class RoutingDecision(BaseModel):
    """Supervisor output when route_mode is 'supervised'."""

    next_agent: str
    reasoning: str = ""
    is_terminal: bool = False


class RecoveryResult(BaseModel):
    recovered: bool
    actions_taken: list[str] = Field(default_factory=list)
    remaining_issues: list[str] = Field(default_factory=list)
    # Some fixes need a human (holding Guide for 2s cannot be automated).
    requires_human: bool = False
    human_instructions: str | None = None
    detail: str = ""
