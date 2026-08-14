"""
recovery_agent.py - agent 9: try to un-stick the rig before giving up.

Disabled by default in agents.yaml. Enable it for attended runs.

WHAT IT CAN AND CANNOT FIX
--------------------------
It can stop stale GIMX processes that are holding the serial port, and it can
release and reopen the capture device. Both are software problems with software
fixes.

It CANNOT authenticate a GIMX session. That needs a human to physically hold
the controller's Guide button for two seconds. No amount of automation replaces
a thumb, which is exactly why the agent reports `requires_human` with printed
instructions rather than pretending to have fixed things.

WHY NOT AUTO-START GIMX
-----------------------
Because starting a session and authenticating it are different acts. An
auto-started server is reachable, looks healthy to every probe, and delivers
nothing. That is strictly worse than no session at all: no session fails
loudly, an unauthenticated one fails silently, and silent failure is what this
whole framework exists to eliminate.

The agent never claims success it cannot verify. After acting it re-probes, and
the graph routes back to the health agent for a genuine re-check rather than
trusting this agent's own optimism.
"""

from __future__ import annotations

import time
from typing import Any

from base import BaseAgent
from schemas import RecoveryResult
from state import AgenticState, note


class RecoveryAgent(BaseAgent):
    """Bounded, honest remediation of rig faults."""

    role = "recovery"
    uses_llm = False

    def run(self, state: AgenticState) -> dict[str, Any]:
        health = state.get("health")
        attempt = int(state.get("recovery_count", 0)) + 1
        actions: list[str] = []
        remaining: list[str] = []
        requires_human = False
        instructions: list[str] = []

        # One attempt only. If releasing the port and reopening the device did
        # not help, doing it again will not either, and each retry costs a
        # minute of a human's attention in an attended run.
        if attempt > 1:
            return self._emit(RecoveryResult(
                recovered=False,
                remaining_issues=["Recovery already attempted once."],
                requires_human=True,
                human_instructions=_MANUAL_STEPS,
                detail="Not retrying: the same automated actions would repeat.",
            ), state, attempt)

        issues = " ".join(health.blocking_issues).lower() if health else ""

        # -- GIMX ----------------------------------------------------------
        if "gimx" in issues or "no gimx session" in issues:
            result = self.call_tool("restart_gimx_session")
            if result.get("ok"):
                actions.append(
                    "Stopped all GIMX processes and released the serial port.")
                requires_human = True
                instructions.append(
                    "Start a session in its own terminal:\n"
                    "    python gimx-session/gimx_session.py start\n"
                    "then HOLD the controller's GUIDE button for 2 SECONDS.\n"
                    "Without that hold the session accepts every event and "
                    "delivers none of them.")
            else:
                remaining.append(f"Could not stop GIMX: {result.get('error')}")

        # -- capture device ------------------------------------------------
        if "capture" in issues or "could not be opened" in issues:
            try:
                # Releasing our own handle is the fix when WE are the second
                # holder. When something else holds it, only closing that app
                # will do, so we say which one.
                self.context.tools.context.hardware.close()
                actions.append("Released our capture handle and reopened it.")
                time.sleep(2.0)
                probe = self.call_tool("check_capture_device")
                if not probe.get("ok") or not probe.get("ready"):
                    conflicts = ", ".join(
                        probe.get("conflicting_apps") or []) or "another app"
                    remaining.append(
                        f"The capture device is still unavailable - "
                        f"{conflicts} is probably holding it.")
                    requires_human = True
                    instructions.append(
                        f"Close {conflicts}. Only one application can hold a "
                        f"capture device at a time.")
            except Exception as exc:
                remaining.append(f"Could not reset capture: {exc}")

        if "not detected" in issues or "signal" in issues:
            # Physical problems. Automation has nothing to offer here, and
            # pretending otherwise just delays the person who has to fix it.
            requires_human = True
            instructions.append(
                "Physical check needed:\n"
                "  1. Is the console powered on and awake (not standby)?\n"
                "  2. Does HDMI run console -> the card's IN port?\n"
                "  3. Is the capture card's USB cable seated?\n"
                "  4. Is the content HDCP-protected (Netflix etc. capture black)?")

        if not actions and not requires_human:
            remaining.append(
                "No automated remedy applies to these issues.")
            requires_human = True
            instructions.append(_MANUAL_STEPS)

        result = RecoveryResult(
            # Deliberately conservative: recovered only when we actually did
            # something AND nothing is left needing hands. The health agent
            # re-probes afterwards and gets the final say.
            recovered=bool(actions) and not remaining and not requires_human,
            actions_taken=actions,
            remaining_issues=remaining,
            requires_human=requires_human,
            human_instructions="\n\n".join(instructions) or None,
            detail=f"Recovery attempt {attempt}.",
        )
        return self._emit(result, state, attempt)

    def _emit(self, result: RecoveryResult, state: AgenticState,
              attempt: int) -> dict[str, Any]:
        if result.requires_human and result.human_instructions:
            # Printed, not just logged. In an attended run somebody is watching
            # the console and can act immediately.
            print("\n" + "=" * 68)
            print("  ACTION REQUIRED - the rig needs a human")
            print("=" * 68)
            print(result.human_instructions)
            print("=" * 68 + "\n")

        self.context.artifacts.save_json(
            f"recovery-{attempt}.json", result.model_dump(mode="json"))

        return {
            "recovery": result.model_dump(mode="json"),
            "recovery_count": attempt,
            "messages": [note(
                self.role,
                f"Recovery {'succeeded' if result.recovered else 'incomplete'}: "
                f"{'; '.join(result.actions_taken) or result.detail}",
                level="info" if result.recovered else "error")],
            "agent_outputs": {self.role: {
                "ok": True,
                "recovered": result.recovered,
                "requires_human": result.requires_human,
            }},
        }


_MANUAL_STEPS = (
    "Manual recovery checklist:\n"
    "  1. python gimx-session/gimx_session.py status\n"
    "  2. python gimx-session/gimx_session.py restart\n"
    "  3. HOLD the controller's GUIDE button for 2 seconds\n"
    "  4. python capture/capture.py preflight\n"
    "  5. Close RECentral 4 / OBS if the capture device will not open"
)
