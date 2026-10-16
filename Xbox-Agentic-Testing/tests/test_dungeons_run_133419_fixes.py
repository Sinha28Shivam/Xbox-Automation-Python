"""Regressions from dungeons-20261016-133419 (60 cycles, navigation worked,
verdict FAIL). Each test names the cycle that showed the problem.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from tools import dungeons_scenario_report as rep
from tools.game_profiles import dungeons_profile as dp
from tools.game_profiles.dungeons_agents import PROFILE
from tools.game_profiles.dungeons_learning import RULES
from tools.game_profiles.dungeons_profile import DungeonsFrameDecision, DungeonsMove

NAV = {"bearing": "left", "shift": 15.0, "occluded": False, "hud": True}


def _hud(health: int, hud: bool = True) -> np.ndarray:
    f = np.full((1080, 1920, 3), 60, dtype=np.uint8)
    if hud:
        x0, y0, x1, y1 = (int(1920 * dp._HUD_HEX_BOX[0]), int(1080 * dp._HUD_HEX_BOX[1]),
                          int(1920 * dp._HUD_HEX_BOX[2]), int(1080 * dp._HUD_HEX_BOX[3]))
        f[y0:y1, x0:x1] = (230, 140, 40)
    hy0, hy1 = int(1080 * dp._HEART_Y[0]), int(1080 * dp._HEART_Y[1])
    hx0, hx1 = int(1920 * dp._HEART_X[0]), int(1920 * dp._HEART_X[1])
    rows = round(dp._HEART_FULL_ROWS * (hy1 - hy0) / 140 * health / 100)
    if rows:
        f[hy1 - rows:hy1, hx0:hx1] = (20, 20, 220)
    return f


def _decide(c: dict[str, Any], entities: list[dict[str, Any]] | None = None,
            to: str = "navigator", **kw: Any):
    d = DungeonsFrameDecision(scene_state="in_mission", player_visible=True, enemies="none",
                              reasoning="t", entities=entities or [], command={"to": to},
                              moves=[DungeonsMove(action="move", direction="left")], **kw)
    d._nav = dict(NAV)
    dp._apply_enemy_geometry(d)
    moves = PROFILE.forced_moves(d, c) or []
    for m in moves:
        PROFILE.on_move_dispatched(m, c)
    return d, moves


# --- 1. story caption (c7-c14) ---------------------------------------------
def test_caption_over_hud_is_advanced_not_walked():
    d = DungeonsFrameDecision(scene_state="in_mission", player_visible=True,
                              enemies="none", reasoning="t", health_percent=100,
                              moves=[DungeonsMove(action="move", direction="up_right")])
    dp._annotate(d, _hud(0, hud=False))
    assert d.scene_state == "menu_or_prompt" and d._nav.get("caption")
    c = PROFILE.initial_counters()
    moves = PROFILE.forced_moves(d, c)
    assert [m.action for m in moves] == ["advance_prompt"] and c["captions_skipped"] == 1
    assert c["_last_health"] == 100            # 0% on a caption is not damage


def test_real_gameplay_frame_is_not_a_caption():
    d = DungeonsFrameDecision(scene_state="in_mission", player_visible=True,
                              enemies="none", reasoning="t", health_percent=100)
    dp._annotate(d, _hud(80))
    assert d.scene_state == "in_mission" and not d._nav.get("caption")
    assert abs(d.health_percent - 80) <= 3


# --- 2. chests: distance from x/y, finish the approach (c42-c60) ----------
CHEST_FAR_SAID_CLOSE = {"category": "loot", "kind": "chest", "x": 0.35, "y": 0.72,
                        "distance_estimate": "very_close"}   # frame 058 geometry


def test_chest_distance_comes_from_position_not_model_claim():
    c = PROFILE.initial_counters()
    d, moves = _decide(c, [dict(CHEST_FAR_SAID_CLOSE)], to="interaction")
    chest = next(e for e in d.entities if e.category == "loot")
    assert chest.distance_estimate == "close" and chest.direction == "down_left"
    assert moves[0].action == "move", moves      # walk, do not press A at range


def test_chest_within_reach_is_opened():
    c = PROFILE.initial_counters()
    d, moves = _decide(c, [{"category": "loot", "kind": "chest", "x": 0.53, "y": 0.56}],
                       to="interaction")
    assert moves[0].action == "interact"


def test_chest_approach_is_not_abandoned_to_navigator():
    c = PROFILE.initial_counters()
    chest = {"category": "loot", "kind": "chest", "x": 0.35, "y": 0.72}
    d, _ = _decide(c, [dict(chest)], to="interaction")
    assert d._agent == "interaction"
    # Next frames: the Observer says navigator (as at c43/c54/c60).
    agents = [_decide(c, [dict(chest)], to="navigator")[0]._agent for _ in range(2)]
    assert agents == ["interaction", "interaction"], agents
    assert c["committed_approaches"] == 2
    # Chest gone from view -> navigator resumes.
    assert _decide(c, [], to="navigator")[0]._agent == "navigator"


# --- 3. ranged fight evidence (c18-c23) ------------------------------------
def test_ranged_fight_counts_when_enemies_cleared_without_damage():
    enemy = {"category": "enemy", "kind": "pillager", "distance_estimate": "far"}
    cycles = [
        {"cycle": 1, "scene_state": "in_mission", "enemy_count": 0, "health_percent": 90,
         "entities": [enemy, enemy], "moves": [{"macro": "ranged", "direction": "right"}]},
        {"cycle": 2, "scene_state": "in_mission", "enemy_count": 0, "health_percent": 90,
         "entities": [], "moves": []},
    ]
    fight = next(s for s in rep.build({"cycles": cycles})["scenarios"]
                 if s["scenario"] == "Fighting mobs")
    assert fight["status"] == "PASS" and fight["evidence_cycles"] == [1]


def test_ranged_fight_not_proven_if_we_lost_health():
    enemy = {"category": "enemy", "kind": "pillager", "distance_estimate": "far"}
    cycles = [
        {"cycle": 1, "scene_state": "in_mission", "health_percent": 90,
         "entities": [enemy], "moves": [{"macro": "ranged", "direction": "right"}]},
        {"cycle": 2, "scene_state": "in_mission", "health_percent": 60, "entities": [],
         "moves": []},
    ]
    fight = next(s for s in rep.build({"cycles": cycles})["scenarios"]
                 if s["scenario"] == "Fighting mobs")
    assert fight["status"] == "FAIL"


# --- 4. potion lesson strength (c19, c54) ----------------------------------
def test_single_death_lesson_stays_mild_even_if_seen_often():
    lesson = {"id": "G001", "failure_class": "death", "occurrences": 3, "quiet_runs": 1,
              "data": {"deaths": 1}}
    adj = RULES.adjustments([lesson])["counters"]
    assert adj["_learn_potion_threshold"] == 80 and adj["_learn_potion_drop"] == 18
    c = {**dp._initial_counters(), "_last_health": 100, **adj}
    quiet = dict(scene_state="in_mission", player_visible=True, enemies="none", reasoning="t")
    assert dp._potion_check(DungeonsFrameDecision(**quiet, health_percent=90), c) is None


def test_repeated_deaths_make_it_strong():
    lesson = {"id": "G001", "failure_class": "death", "quiet_runs": 0, "data": {"deaths": 2}}
    adj = RULES.adjustments([lesson])["counters"]
    assert adj["_learn_potion_threshold"] == 85 and adj["_learn_potion_drop"] == 15


# --- 5. no rescue lesson for a calm villager (G006, c6) --------------------
def test_calm_villager_does_not_record_rescue_lesson():
    villager = {"category": "villager", "kind": "villager", "distance_estimate": "very_close"}
    cycles = [{"cycle": n, "scene_state": "in_mission", "objective_kind": "rescue",
               "villager_in_danger": False, "entities": [villager], "moves": [],
               "agent": "interaction"} for n in range(1, 12)]
    payload = {"cycles": cycles}
    found = RULES.detect(payload, rep.build(payload))
    assert not [f for f in found if f["scenario"] == "Saving villagers"], found
