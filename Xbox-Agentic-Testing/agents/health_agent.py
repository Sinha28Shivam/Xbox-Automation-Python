"""
health_agent.py - agent 1: can this rig run a test at all?

Runs first and has the power to stop everything. Its job is to make BLOCKED a
possible outcome, because the alternative is worse: a rig with a dead capture
card or an unauthenticated GIMX session will happily produce PASS results that
mean nothing. Those false passes are the exact failure this project already
paid for once.

DETERMINISTIC BY DESIGN
-----------------------
No LLM. Every question here is factual - is the device present, does it open,
is the frame flat, is the UDP port answering. A model adds latency, cost and a
chance of hallucination to questions that have exact answers, and it would be
absurd for the component whose whole purpose is trustworthiness to be the one
that guesses. It also means health checks run with no API key configured.

REACHABLE IS NOT AUTHENTICATED
------------------------------
The single most important nuance. GIMX answers UDP as soon as its server is up.
If nobody held the Guide button for 2 seconds, every event is still accepted,
still reports "ok", and still reaches nothing. We cannot detect that from here -
only a screen change proves it - so we record it as a WARNING and leave
`gimx_authenticated` as None. Claiming to know would be a lie; hiding the
question entirely would be worse.
"""

from __future__ import annotations

from typing import Any

from base import BaseAgent
from schemas import ComponentHealth, Evidence, EvidenceKind, HealthReport
from state import AgenticState, note


class HealthAgent(BaseAgent):
    """Probes GIMX, the capture card and the adapters."""

    role = "health"
    uses_llm = False

    def run(self, state: AgenticState) -> dict[str, Any]:
        components: list[ComponentHealth] = []
        evidence: list[Evidence] = []
        blocking: list[str] = []
        warnings: list[str] = []

        adapters_ok = self._check_adapters(components, blocking)
        gimx = self._check_gimx(components, evidence, blocking, warnings)
        capture = self._check_capture(components, evidence, blocking, warnings)

        dry_run = bool(state.get("dry_run", False))
        if dry_run:
            # In dry-run nothing is dispatched and no frames are compared, so
            # hardware faults are irrelevant - but the run also cannot prove
            # anything about the console. Both facts are recorded.
            blocking.clear()
            warnings.append(
                "DRY RUN: no input is sent and no frames are captured. This "
                "run can validate a plan but can never prove console behaviour.")

        healthy = not blocking
        report = HealthReport(
            healthy=healthy,
            components=components,
            blocking_issues=blocking,
            warnings=warnings,
            recoverable=self._is_recoverable(blocking),
            gimx_reachable=bool(gimx.get("reachable")),
            gimx_authenticated=None,      # unknowable without a screen change
            capture_has_signal=bool(capture.get("has_signal")),
            summary=self._summarise(healthy, blocking, warnings, dry_run),
            evidence=evidence,
        )

        self.context.artifacts.save_json(
            "health.json", report.model_dump(mode="json"))

        return {
            "health": report,
            "messages": [note(self.role, report.summary,
                              level="info" if healthy else "error")],
            "agent_outputs": {self.role: {
                "ok": True,
                "healthy": healthy,
                "blocking": blocking,
                "adapters_ok": adapters_ok,
            }},
        }

    # -- probes ------------------------------------------------------------
    def _check_adapters(self, components: list[ComponentHealth],
                        blocking: list[str]) -> bool:
        result = self.call_tool("get_rig_status")
        if not result.get("ok"):
            blocking.append(f"Cannot read rig status: {result.get('error')}")
            return False

        all_ok = True
        for name, info in (result.get("adapters") or {}).items():
            available = bool(info.get("available"))
            all_ok = all_ok and available
            components.append(ComponentHealth(
                name=f"adapter:{name}",
                ok=available,
                detail=info.get("error") or f"loaded from {info.get('path')}",
                metrics={"path": info.get("path")},
                remediation=None if available else (
                    f"Check that {info.get('path')} exists and its Python "
                    f"dependencies are installed (pyyaml, opencv-python)."),
            ))
            if not available:
                blocking.append(
                    f"Hardware adapter '{name}' failed to load: "
                    f"{info.get('error')}")

        if not result.get("controls_exists"):
            blocking.append(
                f"controls.yaml not found at {result.get('controls_config')}. "
                f"Every control name, timing and device setting comes from "
                f"that file - nothing can run without it.")
        return all_ok

    def _check_gimx(self, components: list[ComponentHealth],
                    evidence: list[Evidence], blocking: list[str],
                    warnings: list[str]) -> dict[str, Any]:
        result = self.call_tool("check_gimx_session")
        if not result.get("ok"):
            components.append(ComponentHealth(
                name="gimx", ok=False,
                detail=str(result.get("error")),
                remediation="Start a session: "
                            "python gimx-session/gimx_session.py start"))
            blocking.append(f"GIMX probe failed: {result.get('error')}")
            return {}

        reachable = bool(result.get("reachable"))
        components.append(ComponentHealth(
            name="gimx",
            ok=reachable,
            detail=("GIMX server is reachable over UDP" if reachable
                    else "No GIMX server is answering"),
            metrics={
                "processes": result.get("processes"),
                "udp_address": result.get("udp_address"),
                "serial_port": result.get("serial_port"),
            },
            remediation=None if reachable else (
                "In a separate terminal run:\n"
                "  python gimx-session/gimx_session.py start\n"
                "then HOLD the controller's GUIDE button for 2 seconds."),
        ))
        evidence.append(Evidence(
            kind=EvidenceKind.DEVICE_STATE,
            summary=f"GIMX reachable={reachable}, "
                    f"processes={result.get('processes')}",
            detail=result,
            source_tool="check_gimx_session",
        ))

        if not reachable:
            blocking.append(
                "No GIMX session is reachable, so no input can reach the "
                "console. Nothing can be tested.")
        else:
            # Always warned about, even on a healthy rig. It is invisible,
            # it is the most common cause of a silently-doing-nothing run,
            # and the report should carry the caveat every time.
            warnings.append(
                "GIMX is reachable, but reachable is NOT authenticated. If "
                "nobody held the Guide button for 2s, every event will be "
                "accepted and none will reach the console. Only an observed "
                "screen change settles this.")
        return result

    def _check_capture(self, components: list[ComponentHealth],
                       evidence: list[Evidence], blocking: list[str],
                       warnings: list[str]) -> dict[str, Any]:
        result = self.call_tool("check_capture_device")
        if not result.get("ok"):
            components.append(ComponentHealth(
                name="capture", ok=False, detail=str(result.get("error")),
                remediation="Check the card's USB cable and close RECentral."))
            blocking.append(f"Capture probe failed: {result.get('error')}")
            return {}

        present = bool(result.get("device_present"))
        opens = bool(result.get("device_opens"))
        signal = bool(result.get("has_signal"))

        components.append(ComponentHealth(
            name="capture", ok=bool(result.get("ready")),
            detail="; ".join(result.get("messages") or []) or "capture ready",
            metrics={
                "present": present, "opens": opens, "signal": signal,
                # Recorded every run so a future index shift shows up in the
                # report instead of having to be rediagnosed from scratch.
                "frame_size": result.get("frame_size"),
                "resolved_index": result.get("resolved_index"),
                "index_source": result.get("index_source"),
            },
            remediation=self._capture_remediation(present, opens, signal,
                                                  result.get("conflicting_apps")),
        ))
        evidence.append(Evidence(
            kind=EvidenceKind.DEVICE_STATE,
            summary=f"Capture present={present} opens={opens} signal={signal}",
            detail=result,
            source_tool="check_capture_device",
        ))

        # Each failure mode gets its own message. "Capture not ready" would be
        # true but useless; the fix for a missing device is nothing like the
        # fix for a sleeping console.
        if not present:
            blocking.append(
                "Capture card not detected. Without it nothing can be "
                "verified, and an unverified run cannot pass.")
        elif not opens:
            apps = ", ".join(result.get("conflicting_apps") or []) or "another app"
            blocking.append(
                f"The capture device could not be opened - {apps} is probably "
                f"holding it. Only one application may use a capture device.")
        elif not signal:
            blocking.append(
                "The capture card works but the frame is FLAT: no HDMI signal. "
                "The console may be asleep, the HDMI may be in the wrong port, "
                "or the content may be HDCP-protected.")

        # Wrong-device detection, and it BLOCKS rather than warns. A webcam
        # opens fine and returns dark frames, so testing against it produces
        # confident nonsense - worse than an obvious failure.
        if not result.get("device_identity_ok", True):
            blocking.append(str(result.get("device_warning")))

        if result.get("conflicting_apps") and opens:
            warnings.append(
                f"Running apps that could steal the capture device mid-run: "
                f"{', '.join(result['conflicting_apps'])}")
        return result

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _capture_remediation(present: bool, opens: bool, signal: bool,
                             conflicts: list[str] | None) -> str | None:
        if not present:
            return "Reconnect the capture card's USB cable."
        if not opens:
            return (f"Close {', '.join(conflicts or ['RECentral 4'])} - only "
                    f"one application can hold a capture device.")
        if not signal:
            return ("Wake the console, confirm HDMI goes into the card's IN "
                    "port, and check the content is not HDCP-protected.")
        return None

    @staticmethod
    def _is_recoverable(blocking: list[str]) -> bool:
        """Could a bounded automated retry plausibly fix this?

        Software-held devices and stale processes, yes. Unplugged hardware and
        a sleeping console, no - those need hands. Claiming otherwise just
        burns a retry cycle before failing anyway.
        """
        text = " ".join(blocking).lower()
        recoverable = ("holding it" in text or "could not be opened" in text
                       or "no gimx session" in text)
        hard = "not detected" in text or "adapter" in text
        return recoverable and not hard

    @staticmethod
    def _summarise(healthy: bool, blocking: list[str], warnings: list[str],
                   dry_run: bool) -> str:
        if dry_run:
            return ("DRY RUN: hardware checks bypassed. Plans can be validated "
                    "but no console behaviour can be proven.")
        if healthy:
            base = "Rig is ready: GIMX reachable and a live video signal present."
            return f"{base} {len(warnings)} caveat(s) recorded." if warnings else base
        return (f"Rig is NOT usable - {len(blocking)} blocking issue(s). "
                f"Result will be BLOCKED, not FAIL: with the rig in this state "
                f"nothing can be concluded about the console.")
