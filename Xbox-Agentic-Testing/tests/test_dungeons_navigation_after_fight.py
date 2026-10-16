"""After a fight the hero must follow the minimap to the objective.

Regression for dungeons-20261016-130821: cycles 16-60 were 45/45 combat,
38 of them with every enemy 'far' (often phantoms at the screen edges), so
the navigator never ran and the hero never reached the yellow objective.
"""
from __future__ import annotations

from typing import Any

from tools.game_profiles import dungeons_profile as dp
from tools.game_profiles.dungeons_agents import PROFILE
from tools.game_profiles.dungeons_profile import DungeonsFrameDecision, DungeonsMove

NAV = {"bearing": "up_right", "shift": 15.0, "occluded": False, "hud": True}


def _decide(c: dict[str, Any], enemies: list[tuple[float, float]] | None = None,
            hp: int = 100, entities: list[dict[str, Any]] | None = None,
            to: str = "combat") -> tuple[DungeonsFrameDecision, list[DungeonsMove]]:
    ents = [{"category": "enemy", "kind": "pillager", "x": x, "y": y}
            for x, y in (enemies or [])] + list(entities or [])
    d = DungeonsFrameDecision(scene_state="in_mission", player_visible=True,
                              enemies="pillager" if enemies else "none", reasoning="t",
                              health_percent=hp, entities=ents,
                              command={"to": to}, moves=[DungeonsMove(action="move",
                                                                      direction="up_right")])
    d._nav = dict(NAV)
    dp._apply_enemy_geometry(d)
    moves = PROFILE.forced_moves(d, c) or []
    for m in moves:
        PROFILE.on_move_dispatched(m, c)
    return d, moves


def test_far_enemies_do_not_stop_navigation():
    c = PROFILE.initial_counters()
    # A pillager at the top-left edge, far from the hero, frame after frame.
    agents = [_decide(c, enemies=[(0.15, 0.15)])[0]._agent for _ in range(6)]
    assert agents == ["navigator"] * 6, agents
    assert c["far_ignored"] == 6 and c["engagements"] == 0


def test_far_enemy_fights_back_after_damage():
    c = PROFILE.initial_counters()
    _decide(c, enemies=[(0.15, 0.15)], hp=100)
    d, moves = _decide(c, enemies=[(0.15, 0.15)], hp=85)   # hit: it is real
    assert d._agent == "combat" and any(m.action == "ranged" for m in moves)


def test_close_enemy_still_fights():
    c = PROFILE.initial_counters()
    d, moves = _decide(c, enemies=[(0.58, 0.48)])           # adjacent, right
    assert d._agent == "combat" and moves[-1].action == "fight"


def test_ranged_standoff_pushes_to_objective():
    c = PROFILE.initial_counters()
    # 'close' (inside _NEAR_RADIUS) but not adjacent: a ranged exchange.
    seq = []
    for _ in range(6):
        _, moves = _decide(c, enemies=[(0.62, 0.40)])
        seq.append((moves, c["_sustain_action"]))
    pushes = [(moves, sus) for moves, sus in seq
              if any("push to objective" in m.purpose for m in moves)]
    assert pushes, [[m.action for m in mv] for mv, _ in seq]
    moves, sustain_after = pushes[0]
    assert moves[-1].action == "move" and moves[-1].direction == "up_right"
    assert sustain_after is None               # sustain will not keep shooting


def test_coins_are_walked_over_not_pressed():
    c = PROFILE.initial_counters()
    d, moves = _decide(c, to="interaction", entities=[
        {"category": "loot", "kind": "glowing gold coins", "direction": "left",
         "distance_estimate": "close"}])
    assert d._agent == "interaction"
    assert moves[0].action == "move" and moves[0].direction == "left"
    assert c["coin_pickups"] == 1


def test_chest_still_opened_with_a():
    c = PROFILE.initial_counters()
    d, moves = _decide(c, to="interaction", entities=[
        {"category": "loot", "kind": "chest", "direction": "up",
         "distance_estimate": "very_close"}])
    assert d._agent == "interaction" and moves[0].action == "interact"


def test_report_counts_spilled_coins_as_looted():
    from tools import dungeons_scenario_report as rep
    chest = {"category": "loot", "kind": "chest", "distance_estimate": "very_close"}
    coins = {"category": "loot", "kind": "glowing coins", "distance_estimate": "close"}
    cycles = [
        {"cycle": 1, "scene_state": "in_mission", "entities": [chest],
         "moves": [{"macro": "interact", "purpose": "Interact with loot 'chest'."}]},
        {"cycle": 2, "scene_state": "in_mission", "entities": [chest, coins], "moves": []},
    ]
    report = rep.build({"cycles": cycles})
    loot = next(s for s in report["scenarios"] if s["scenario"] == "Looting chests")
    assert loot["status"] == "PASS" and loot["evidence_cycles"] == [1]
