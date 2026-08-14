"""
supervisor_agent.py - agent 8: dynamic routing (optional).

Only used when graph.yaml sets `route_mode: supervised`. In `static` mode the
declared edges run the workflow and this agent never executes.

WHY IT IS OPTIONAL AND OFF BY DEFAULT
-------------------------------------
The health -> validate -> plan -> execute -> verify -> report pipeline is a
genuinely good default for this domain, and a static graph is cheaper, faster
and reproducible. Asking a model "what next?" after every node adds a call each
time and makes two runs of the same scenario take different paths, which is a
poor property for a test framework.

It exists for the cases the fixed pipeline cannot express: skipping planning
when the scenario is already a literal step list, looping verification over a
long game load, running RCA before a retry. Those are real, so the capability
is here - just not on by default.

BOUNDED BY CONSTRUCTION
-----------------------
Step and replan caps are enforced in code after the model answers. A supervisor
that can route freely can also route in circles, and this one is driving real
hardware.
"""

from __future__ import annotations

from typing import Any

from base import BaseAgent
from schemas import RoutingDecision
from state import AgenticState, note


class SupervisorAgent(BaseAgent):
    """Chooses the next agent to run, within hard limits."""

    role = "supervisor"

    def run(self, state: AgenticState) -> dict[str, Any]:
        forced = self._forced_route(state)
        if forced is not None:
            return self._emit(forced, state)

        available = [
            role for role, spec in
            (self.context.agents_config.section("agents") or {}).items()
            if spec.get("enabled", True) and role != self.role
        ]

        prompt = self.render_prompt(
            state,
            available_agents=available,
            completed_agents=sorted(state.get("agent_outputs", {})),
            max_steps=self.context.settings.get("runtime.max_steps", 40),
            max_replans=self.context.settings.get("runtime.max_replans", 3),
        )

        decision = self.invoke_structured(RoutingDecision, prompt)

        # Never trust a routed name blindly: a hallucinated agent id would
        # crash the graph, and "reporter" is always a safe terminal.
        if decision.next_agent not in available:
            decision = RoutingDecision(
                next_agent="reporter",
                reasoning=(f"'{decision.next_agent}' is not a known agent; "
                           f"routing to the reporter instead."),
                is_terminal=True,
            )
        return self._emit(decision, state)

    # -- hard limits -------------------------------------------------------
    def _forced_route(self, state: AgenticState) -> RoutingDecision | None:
        """Situations where the model does not get a say."""
        if state.get("should_stop"):
            return RoutingDecision(
                next_agent="reporter", is_terminal=True,
                reasoning=f"Run halted: {state.get('stop_reason', '')}")

        if int(state.get("step_count", 0)) >= int(
                self.context.settings.get("runtime.max_steps", 40)):
            return RoutingDecision(
                next_agent="reporter", is_terminal=True,
                reasoning="Step budget exhausted; reporting what we have.")

        if int(state.get("replan_count", 0)) > int(
                self.context.settings.get("runtime.max_replans", 3)):
            return RoutingDecision(
                next_agent="reporter", is_terminal=True,
                reasoning="Replan limit reached; further attempts are unlikely "
                          "to differ.")

        health = state.get("health")
        if health is not None and not health.healthy:
            return RoutingDecision(
                next_agent="reporter", is_terminal=True,
                reasoning="The rig is unusable; the only honest outcome is "
                          "BLOCKED.")
        return None

    def _emit(self, decision: RoutingDecision,
              state: AgenticState) -> dict[str, Any]:
        return {
            "next_agent": decision.next_agent,
            "messages": [note(
                self.role,
                f"-> {decision.next_agent}: {decision.reasoning}")],
            "agent_outputs": {self.role: {
                "ok": True,
                "next_agent": decision.next_agent,
                "terminal": decision.is_terminal,
            }},
        }
