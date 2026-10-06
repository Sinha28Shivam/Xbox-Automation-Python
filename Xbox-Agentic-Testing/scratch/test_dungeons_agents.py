"""Offline check: the supervisor hands each situation to the right agent."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for sub in ("core", "tools", "agents", "graph", "tools/game_profiles"):
    sys.path.insert(0, str(ROOT / sub))

from game_profiles.dungeons_agents import PROFILE
from game_profiles.dungeons_profile import DungeonsFrameDecision, DungeonsMove

Q = dict(scene_state="in_mission", player_visible=True, enemies="none", reasoning="t",
         moves=[DungeonsMove(action="move", direction="right", duration=2.0)])
MOVING = {"bearing": "up_right", "shift": 30.0, "occluded": False}


def run(c, **kw):
    d = DungeonsFrameDecision(**{**Q, **kw})
    d._nav = dict(MOVING)
    moves = PROFILE.forced_moves(d, c)
    for m in moves or d.moves:
        PROFILE.on_move_dispatched(m, c)
    return d._agent, [(m.action, m.direction) for m in moves or []]


def ent(cat, kind, direction="up", dist="close", **kw):
    return dict(category=cat, kind=kind, direction=direction, distance_estimate=dist, **kw)


c = PROFILE.initial_counters()
a, mv = run(c)
assert a == "navigator" and mv == [("move", "up_right")], (a, mv)  # follows minimap
assert run(c, health_percent=40)[0] == "health"
c = PROFILE.initial_counters()
a, mv = run(c, enemies="zombie", enemy_adjacent=True, enemy_direction="left")
assert a == "combat" and mv[-1] == ("fight", "left"), (a, mv)
# Low health beats an adjacent enemy.
c = PROFILE.initial_counters()
assert run(c, enemy_adjacent=True, enemy_direction="left", health_percent=30)[0] == "health"

# Chest close -> approach, very close -> A, gives up after 3 presses.
TO_INTERACT = dict(to="interaction", target="chest")
TO_COMBAT = dict(to="combat")
c = PROFILE.initial_counters()
a, mv = run(c, command=TO_INTERACT, entities=[ent("loot", "chest", "up_right")])
assert a == "interaction" and mv == [("move", "up_right")], (a, mv)
presses = [run(c, command=TO_INTERACT, entities=[ent("loot", "chest", "up", "very_close")])
           for _ in range(4)]
assert all(p == ("interaction", [("interact", "up")]) for p in presses[:3]), presses
assert presses[3][0] == "navigator", presses[3]
assert c["interaction_give_ups"] == 1, c

# Observer sends interaction while a distant enemy is visible: interaction acts.
c = PROFILE.initial_counters()
a, mv = run(c, command=dict(to="interaction", target="villager"), enemy_direction="down",
            entities=[ent("villager", "villager", "left", "very_close")])
assert a == "interaction" and mv == [("interact", "left")], (a, mv)
# ...but an adjacent enemy overrides the command.
a, _ = run(c, command=TO_INTERACT, enemy_adjacent=True, enemy_direction="down",
           entities=[ent("loot", "chest", "left", "very_close")])
assert a == "combat", a

# Enemy attacking a villager is targeted over the nearest enemy.
c = PROFILE.initial_counters()
a, mv = run(c, enemy_direction="down", villager_in_danger=True,
            entities=[ent("enemy", "zombie", "left", threatens_villager=True)])
assert a == "combat" and mv == [("fight", "left")], (a, mv)

# Ranged enemy only -> combat shoots.
c = PROFILE.initial_counters()
a, mv = run(c, command=TO_COMBAT, enemy_direction="up")
assert a == "combat" and ("ranged", "up") in mv, (a, mv)

# Downed -> scene handler.
assert run(c, scene_state="death")[0] == "scene"
s = PROFILE.summarize(c)
assert "agent_cycles" in s, s
print("agents: OK", s["agent_cycles"])
