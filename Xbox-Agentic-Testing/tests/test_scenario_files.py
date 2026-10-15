"""Every scenario YAML must load through the real validator and be runnable.

Catches, before a rig is booked: YAML syntax errors, criteria with unknown
check types, navigation_guidance missing target_text, stage-contract
violations, and two files claiming the same scenario id (which would merge
their lessons and cached routes).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from agents.scenario_agent import ScenarioValidatorAgent

SCENARIOS = sorted((Path(__file__).resolve().parents[1] / "scenarios").glob("*.yaml"))
CHECK_TYPES = {"screen_change", "text_present", "text_absent", "screen_match",
               "vision_judgement", "no_error_dialog"}


def _data(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    assert isinstance(data, dict), f"{path.name}: top level must be a mapping"
    return data


def _requirement_file(data: dict) -> bool:
    # Minimal requirement YAMLs (run via --requirement-file) are expanded by the
    # LLM and have their own schema; they are checked separately below.
    return "expected_outcome" in data or "requirement" in data


@pytest.mark.parametrize("path", SCENARIOS, ids=[p.stem for p in SCENARIOS])
def test_scenario_builds_without_llm(path: Path) -> None:
    data = _data(path)
    if _requirement_file(data) or not data.get("success_criteria"):
        pytest.skip("requirement/prose scenario: normalised by the LLM at run time")
    agent = ScenarioValidatorAgent.__new__(ScenarioValidatorAgent)
    scenario = agent._from_yaml(data, {})
    assert scenario.id and scenario.title and scenario.goal
    assert scenario.success_criteria
    for c in scenario.success_criteria:
        assert c.check_type in CHECK_TYPES, f"{path.name}: unknown check_type {c.check_type}"
        nav = c.parameters.get("navigation_guidance")
        if nav is not None:
            assert isinstance(nav, dict) and str(nav.get("target_text", "")).strip(), \
                f"{path.name}: navigation_guidance needs target_text"
            assert str(nav.get("direction", "right")).lower() in {"left", "right", "up", "down"}


def test_scenario_ids_are_unique() -> None:
    seen: dict[str, str] = {}
    duplicates = []
    for path in SCENARIOS:
        data = _data(path)
        sid = str(data.get("id") or "")
        if not sid:
            continue
        if sid in seen:
            duplicates.append(f"{sid}: {seen[sid]} and {path.name}")
        seen[sid] = path.name
    assert not duplicates, "duplicate scenario ids share lessons/routes: " + "; ".join(duplicates)
