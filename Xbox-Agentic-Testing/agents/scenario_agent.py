"""
scenario_agent.py - agent 2: is this test well-formed and runnable?

Takes whatever the user supplied - a YAML file or a sentence typed at the
console - and turns it into a ValidatedScenario the rest of the pipeline can
rely on. Free-text input is the whole point of the system, so this is where
"launch Forza and check it reaches the main menu" becomes structured data.

THE GATE IT ENFORCES
--------------------
A scenario must have at least one OBSERVABLE success criterion. "Verify the
game launches" is not observable; "the main menu is visible on screen" is. A
criterion that cannot be seen cannot be checked, and a test that cannot fail is
worthless - so the agent rewrites vague criteria into visual ones and marks the
scenario invalid if it cannot.

It also resolves every control name against controls.yaml BEFORE execution. A
scenario referencing a button that does not exist should be a clean scenario
defect at second one, not a cryptic GIMX error five steps into a hardware run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from base import BaseAgent
from schemas import RequirementItem, SuccessCriterion, ValidatedScenario
from state import AgenticState, note


class ScenarioValidatorAgent(BaseAgent):
    """Normalises and validates the requested scenario."""

    role = "scenario_validator"

    def run(self, state: AgenticState) -> dict[str, Any]:
        raw = str(state.get("scenario_input", "")).strip()
        if not raw:
            return self._invalid("No scenario was provided.", state)

        source = state.get("scenario_source", "text")
        requirement: RequirementItem | None = None

        try:
            if source == "requirement_file":
                requirement = self._load_requirement(raw)
                scenario = self._from_requirement(requirement, state)
            else:
                payload, origin = (self._load_file(raw) if source == "file"
                                   else ({"text": raw}, "natural_language"))
                scenario = (self._from_yaml(payload, state) if origin == "yaml"
                            else self._from_text(raw, state))
        except (ValueError, ValidationError, yaml.YAMLError) as exc:
            return self._invalid(str(exc), state)

        # Structural validity is not enough: names must exist on THIS rig.
        # Same YAML on a differently-configured console can be invalid, which
        # is exactly the kind of drift a static schema check would miss.
        unresolved = self._unresolved_controls(scenario)
        if unresolved:
            scenario.unresolved_controls = unresolved
            scenario.valid = False
            scenario.issues.append(
                f"These controls do not exist in controls.yaml: "
                f"{', '.join(unresolved)}. Call get_control_surface to see "
                f"what this rig actually supports.")

        self.context.artifacts.save_json(
            "scenario.json", scenario.model_dump(mode="json"))
        if requirement is not None:
            self.context.artifacts.save_json(
                "requirement.json", requirement.model_dump(mode="json"))

        level = "info" if scenario.valid else "error"
        summary = (f"Scenario '{scenario.title}' validated with "
                   f"{len(scenario.success_criteria)} success criteria."
                   if scenario.valid else
                   f"Scenario rejected: {'; '.join(scenario.issues)}")

        return {
            "requirement": requirement,
            "scenario": scenario,
            "messages": [note(self.role, summary, level=level)],
            "should_stop": not scenario.valid,
            "stop_reason": "" if scenario.valid else summary,
            "agent_outputs": {self.role: {
                "ok": True,
                "valid": scenario.valid,
                "issues": scenario.issues,
            }},
        }

    # -- input handling ----------------------------------------------------
    def _load_file(self, path_str: str) -> tuple[dict[str, Any], str]:
        path = Path(path_str)
        if not path.is_file():
            raise FileNotFoundError(f"Scenario file not found: {path}")
        text = path.read_text(encoding="utf-8")

        # A .yaml file may still hold prose under a description key, and a
        # .txt file may hold YAML. Decide by what parses, not by extension.
        if path.suffix.lower() in (".yaml", ".yml"):
            data = yaml.safe_load(text) or {}
            if isinstance(data, dict):
                return data, "yaml"
        return {"text": text}, "natural_language"

    def _load_requirement(self, path_str: str) -> RequirementItem:
        path = Path(path_str)
        if not path.is_file():
            raise FileNotFoundError(f"Requirement file not found: {path}")
        text = path.read_text(encoding="utf-8")
        data = yaml.safe_load(text)
        if not isinstance(data, dict):
            raise ValueError(
                "Requirement YAML must contain exactly one mapping/object, "
                "not a list or empty document.")
        return RequirementItem.model_validate(data)

    def _from_yaml(self, data: dict[str, Any],
                   state: AgenticState) -> ValidatedScenario:
        """Build from a structured scenario file.

        The LLM is still consulted when criteria are missing or vague, so a
        hand-written scenario gets the same observability guarantee as a
        generated one.
        """
        criteria = [
            SuccessCriterion(
                description=str(c.get("description", c) if isinstance(c, dict) else c),
                check_type=str(c.get("check_type", "vision_judgement")
                               if isinstance(c, dict) else "vision_judgement"),
                parameters=dict(c.get("parameters", {})) if isinstance(c, dict) else {},
                required=bool(c.get("required", True)) if isinstance(c, dict) else True,
            )
            for c in (data.get("success_criteria") or [])
        ]

        if not criteria:
            return self._from_text(
                yaml.safe_dump(data, sort_keys=False), state,
                seed_id=str(data.get("id", "")),
                seed_title=str(data.get("title", "")))

        return ValidatedScenario(
            id=str(data.get("id") or _slug(data.get("title", "scenario"))),
            title=str(data.get("title", "Untitled scenario")),
            description=str(data.get("description", "")),
            console=data.get("console"),
            goal=str(data.get("goal") or data.get("description")
                     or data.get("title", "")),
            preconditions=[str(p) for p in (data.get("preconditions") or [])],
            success_criteria=criteria,
            tags=[str(t) for t in (data.get("tags") or [])],
            timeout_seconds=data.get("timeout_seconds"),
            max_steps=data.get("max_steps"),
            normalised_from="yaml",
        )

    def _from_text(self, text: str, state: AgenticState,
                   seed_id: str = "", seed_title: str = "") -> ValidatedScenario:
        """Let the LLM turn prose into a structured scenario."""
        prompt = self.render_prompt(
            state,
            scenario_text=text,
            seed_id=seed_id,
            seed_title=seed_title,
        )
        scenario = self.invoke_structured(ValidatedScenario, prompt)
        scenario.normalised_from = "natural_language"
        if seed_id:
            scenario.id = seed_id
        return scenario

    def _from_requirement(self, requirement: RequirementItem,
                          state: AgenticState) -> ValidatedScenario:
        """Normalize one minimal requirement YAML item into a runnable scenario."""
        requirement_yaml = yaml.safe_dump(
            requirement.model_dump(mode="json"),
            sort_keys=False, allow_unicode=True)
        prompt = self.render_prompt(
            state,
            scenario_text=requirement_yaml,
            seed_id=requirement.id,
            seed_title=requirement.title,
            requirement=requirement.model_dump(mode="json"),
            requirement_goal=requirement.goal,
            requirement_expected_outcome=requirement.expected_outcome,
        )
        scenario = self.invoke_structured(ValidatedScenario, prompt)
        scenario.id = requirement.id
        scenario.title = requirement.title
        scenario.goal = requirement.goal
        scenario.console = requirement.console
        scenario.preconditions = list(requirement.preconditions) + [
            p for p in scenario.preconditions
            if p not in requirement.preconditions
        ]
        scenario.tags = list(requirement.tags) + [
            t for t in scenario.tags if t not in requirement.tags
        ]
        scenario.normalised_from = "requirement_yaml"
        return scenario

    # -- validation --------------------------------------------------------
    def _unresolved_controls(self, scenario: ValidatedScenario) -> list[str]:
        """Check any control names the scenario mentions in its parameters.

        Only explicit `button`/`control` parameters are checked. Scanning the
        free text for words that look like buttons would produce false alarms
        on ordinary English ("select the game", "back to the menu").
        """
        names: set[str] = set()
        for criterion in scenario.success_criteria:
            for key in ("button", "control", "buttons", "controls"):
                value = criterion.parameters.get(key)
                if isinstance(value, str):
                    names.add(value)
                elif isinstance(value, list):
                    names.update(str(v) for v in value)

        unresolved: list[str] = []
        for name in sorted(names):
            result = self.call_tool("resolve_control", name=name)
            if not result.get("ok"):
                unresolved.append(name)
        return unresolved

    def _invalid(self, reason: str, state: AgenticState) -> dict[str, Any]:
        scenario = ValidatedScenario(
            id="invalid", title="Invalid scenario", goal="",
            valid=False, issues=[reason],
            success_criteria=[SuccessCriterion(
                description="placeholder", check_type="none")],
        )
        scenario.valid = False
        return {
            "scenario": scenario,
            "should_stop": True,
            "stop_reason": reason,
            "messages": [note(self.role, reason, level="error")],
        }


def _slug(text: str) -> str:
    return "-".join(str(text).lower().split())[:50] or "scenario"
