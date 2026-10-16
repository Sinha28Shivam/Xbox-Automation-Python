"""Dungeons gameplay learning + report, replayed from a real run.

Fixture: tests/fixtures/dungeons/dungeons-20261016-114012-cycles.json, the
trimmed cycle log of a 60-cycle run on the rig (1 death at c37, chests seen
close at c1/c40/c42 but never opened).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from core import gameplay_lessons as gl
from core.professional_report import render_dashboard
from tools import dungeons_scenario_report as rep
from tools.game_profiles import dungeons_profile as dp
from tools.game_profiles.dungeons_agents import PROFILE
from tools.game_profiles.dungeons_learning import RULES, death_cycles
from tools.game_profiles.dungeons_profile import DungeonsFrameDecision, DungeonsMove

FIXTURE = (Path(__file__).parent / "fixtures" / "dungeons"
           / "dungeons-20261016-114012-cycles.json")


@pytest.fixture()
def payload() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _scen(report: dict, name: str) -> dict:
    return next(s for s in report["scenarios"] if s["scenario"] == name)


# --- report ----------------------------------------------------------------
def test_objective_title_suffix_is_not_a_new_stage(payload):
    # "Defeat the Illagers from the Rift" at c42 used to normalise to "defeat"
    # and count as objective progress.
    stages = rep.objective_stages([c for c in payload["cycles"]
                                   if c["scene_state"] == "in_mission"])
    assert len(stages) == 1, stages
    assert _scen(rep.build(payload), "Objective reached")["status"] == "OBSERVED"


def test_objective_genuine_change_still_passes():
    cycles = [{"cycle": n, "scene_state": "in_mission", "objective": o}
              for n, o in [(1, "Find the Farmer"), (2, "Find the Farmer"),
                           (3, "Defeat the Illagers"), (4, "Defeat the Illagers")]]
    assert [s for _, s in rep.objective_stages(cycles)] == ["find the farmer",
                                                            "defeat the illagers"]


def test_report_has_counts_summary_moments_timeline(payload):
    r = rep.build(payload)
    assert r["counts"] == {"NOT ENCOUNTERED": 2, "PASS": 4, "OBSERVED": 3}
    assert "Deaths 1" in r["summary"] and "never moved to a new stage" in r["summary"]
    kinds = [(m["cycle"], m["kind"]) for m in r["moments"]]
    assert (37, "death") in kinds and (45, "potion") in kinds
    assert len(r["timeline"]) == len(payload["cycles"])


def test_dashboard_shows_real_counts_and_sections(payload, tmp_path):
    payload["learning"] = {"enabled": True, "applied": [], "enforced": {},
                           "lessons": {"recorded": [{"id": "G001", "scenario": "Survival",
                                                     "primary_cause": "downed at c37"}]}}
    r = rep.build(payload)
    page = render_dashboard({**r, "run_id": "x", "run_dir": str(tmp_path)}, mode="dungeons")
    # The old dashboard read no counts and showed Passed 0 for a 4-PASS run.
    assert "<strong>4</strong>" in page
    for section in ("Run summary", "Learning", "Key moments", "Cycle timeline", "G001"):
        assert section in page, section


def test_combat_speed_baseline(payload, tmp_path):
    sp = rep.build(payload)["combat_speed"]
    assert sp["combat_cycles"] == 18 and sp["avg_combat_think_s"] == 5.56
    assert sp["attacks_per_combat_cycle"] == 0.94 and sp["sustain_seconds"] == 0
    page = render_dashboard({**rep.build(payload), "run_dir": str(tmp_path)}, mode="dungeons")
    assert "Combat speed" in page and "5.56s" in page


def test_write_keeps_old_report_name(payload, tmp_path):
    rep.write(payload, tmp_path / "reports")
    for name in ("execution-report.html", "dungeons-scenarios.html",
                 "dungeons-scenarios-raw.html", "dungeons-scenarios.json"):
        assert (tmp_path / "reports" / name).is_file(), name


# --- lessons ---------------------------------------------------------------
def test_real_run_records_death_and_skipped_chest(payload):
    found = {(f["rule"], f["scenario"]) for f in RULES.detect(payload, rep.build(payload))}
    assert ("death", "Survival") in found
    assert ("not_attempted", "Looting chests") in found
    assert death_cycles(payload) == [37]


def test_lesson_lifecycle_record_recur_resolve(payload, tmp_path):
    report = rep.build(payload)
    first = gl.update(tmp_path, "dungeons", RULES, payload, report, [], run_id="r1")
    assert len(first["recorded"]) == 3 and not first["recurred"]
    ids = [l["id"] for l in gl.open_lessons(tmp_path, "dungeons")]

    again = gl.update(tmp_path, "dungeons", RULES, payload, report, ids, run_id="r2")
    assert len(again["recurred"]) == 3 and not again["recorded"]
    death = next(l for l in gl.open_lessons(tmp_path, "dungeons")
                 if l["failure_class"] == "death")
    assert death["occurrences"] == 2

    # A later run with no death that fought, and that opened the chest.
    clean = json.loads(json.dumps(payload))
    for c in clean["cycles"]:
        if c["scene_state"] == "death":
            c["scene_state"] = "in_mission"
    fixed = {**report, "scenarios": [
        {**s, "status": "PASS"} if s["scenario"] in ("Looting chests", "Interacting with items")
        else s for s in report["scenarios"]]}
    third = gl.update(tmp_path, "dungeons", RULES, clean, fixed, ids, run_id="r3")
    assert len(third["resolved"]) == 2          # chest + item lessons
    assert third["quiet"] == [death["id"]]      # death ages, not resolved by 1 run
    left = gl.open_lessons(tmp_path, "dungeons")
    assert [l["failure_class"] for l in left] == ["death"]


def test_quiet_lesson_expires(payload, tmp_path):
    gl.update(tmp_path, "dungeons", RULES, payload, rep.build(payload), [], run_id="r1")
    death = next(l for l in gl.open_lessons(tmp_path, "dungeons")
                 if l["failure_class"] == "death")
    clean = json.loads(json.dumps(payload))
    for c in clean["cycles"]:
        if c["scene_state"] == "death":
            c["scene_state"] = "in_mission"
    report = rep.build(clean)
    for n in range(3):
        out = gl.update(tmp_path, "dungeons", RULES, clean, report, [death["id"]],
                        run_id=f"q{n}", expire_after=3)
    assert death["id"] in out["expired"]


def test_short_run_does_not_judge_lessons(payload, tmp_path):
    gl.update(tmp_path, "dungeons", RULES, payload, rep.build(payload), [], run_id="r1")
    ids = [l["id"] for l in gl.open_lessons(tmp_path, "dungeons")]
    short = {**payload, "cycles": payload["cycles"][:5]}
    out = gl.update(tmp_path, "dungeons", RULES, short, rep.build(short), ids, run_id="r2")
    # The chest skipped at c1/c2 is skipped again (a real recurrence); the
    # death lesson is neither aged nor resolved by 5 cycles.
    assert not out["resolved"] and not out["quiet"]
    death = next(l["id"] for l in gl.open_lessons(tmp_path, "dungeons")
                 if l["failure_class"] == "death")
    assert out["untested"] == [death] and len(out["recurred"]) == 2


def test_store_is_listed_by_lessons_cli(payload, tmp_path):
    from core import lesson_store
    gl.update(tmp_path, "dungeons", RULES, payload, rep.build(payload), [], run_id="r1")
    listed = lesson_store.all_lessons(tmp_path)
    assert "gameplay-dungeons" in listed and len(listed["gameplay-dungeons"]) == 3


# --- runtime enforcement -----------------------------------------------------
def test_death_lesson_drinks_earlier():
    adj = RULES.adjustments([{"id": "G1", "failure_class": "death", "occurrences": 1}])
    assert adj["counters"]["_learn_potion_threshold"] == 80
    quiet = dict(scene_state="in_mission", player_visible=True, enemies="none",
                 reasoning="t", health_percent=78)
    # 82% -> 78%: below neither default trigger (70% / -20%).
    base = {**dp._initial_counters(), "_last_health": 82}
    assert dp._potion_check(DungeonsFrameDecision(**quiet), base) is None
    learned = {**dp._initial_counters(), "_last_health": 82, **adj["counters"]}
    moves = dp._potion_check(DungeonsFrameDecision(**quiet), learned)
    assert moves and moves[0].action == "potion" and "[learned" in moves[0].purpose
    assert learned["learn_early_potions"] == 1


def test_interaction_lesson_opens_close_chest_navigator_skipped():
    adj = RULES.adjustments([{"id": "G2", "failure_class": "not_attempted",
                              "data": {"categories": ["loot"]}}])

    def decide(counters):
        d = DungeonsFrameDecision(
            scene_state="in_mission", player_visible=True, enemies="none", reasoning="t",
            command={"to": "navigator", "target": "objective", "direction": "up"},
            entities=[{"category": "loot", "kind": "glowing chest", "direction": "left",
                       "distance_estimate": "very_close"}],
            moves=[DungeonsMove(action="move", direction="up")])
        d._nav = {"bearing": "up", "shift": 30.0, "occluded": False}
        return d, PROFILE.forced_moves(d, counters)

    d, moves = decide(PROFILE.initial_counters())
    assert d._agent == "navigator"               # baseline: walks past the chest
    c = {**PROFILE.initial_counters(), **adj["counters"]}
    d, moves = decide(c)
    assert d._agent == "interaction" and moves[0].action == "interact"
    assert "[learned]" in moves[0].purpose and c["learn_interact_overrides"] == 1


def test_lesson_prompt_reaches_model_session_text():
    lessons = [{"id": "G1", "failure_class": "death", "primary_cause": "downed at c37"}]
    c = {**PROFILE.initial_counters(), "_learn_prompt": RULES.prompt_text(lessons)}
    text = PROFILE.extra_session_text(c)
    assert "LEARNED FROM EARLIER RUNS" in text and "G1" in text
