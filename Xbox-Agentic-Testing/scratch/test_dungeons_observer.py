"""Offline check: Observer command -> supervisor -> action agent, and report rules."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for sub in ("core", "tools", "agents", "graph", "tools/game_profiles"):
    sys.path.insert(0, str(ROOT / sub))

import dungeons_scenario_report as rep
from game_profiles.dungeons_agents import PROFILE
from game_profiles.dungeons_profile import DungeonsFrameDecision

Q = dict(scene_state="in_mission", player_visible=True, enemies="none", reasoning="t")
MOVING = {"bearing": "up_right", "shift": 30.0, "occluded": False}
STILL = {"bearing": "up_right", "shift": 0.5, "occluded": False}


def run(c, nav=MOVING, **kw):
    d = DungeonsFrameDecision(**{**Q, **kw})
    d._nav = dict(nav)
    moves = PROFILE.forced_moves(d, c)
    for m in moves or []:
        PROFILE.on_move_dispatched(m, c)
    return d._agent, d._route, [(m.action, m.direction) for m in moves or []]


def ent(cat, kind, direction="up", dist="close", **kw):
    return dict(category=cat, kind=kind, direction=direction, distance_estimate=dist, **kw)


def cmd(to, target="", direction="none"):
    return dict(to=to, target=target, direction=direction, reason="t")


# Command is delivered to the named agent.
c = PROFILE.initial_counters()
a, r, mv = run(c, command=cmd("interaction", "chest", "left"),
               entities=[ent("loot", "chest", "left", "very_close")])
assert (a, r, mv) == ("interaction", "command", [("interact", "left")]), (a, r, mv)

# Navigator always moves, even with no model moves; minimap beats a bad hint.
c = PROFILE.initial_counters()
a, r, mv = run(c, command=cmd("navigator", "objective", "down_left"))
assert a == "navigator" and mv == [("move", "up_right")], (a, mv)
a, r, mv = run(c, command=cmd("navigator", "objective", "right"))
assert mv == [("move", "right")], mv          # within 90 deg: Observer hint kept

# Safety overrides beat the command.
c = PROFILE.initial_counters()
a, r, _ = run(c, command=cmd("navigator"), health_percent=40)
assert a == "health" and r.startswith("override"), (a, r)
c = PROFILE.initial_counters()
a, r, mv = run(c, command=cmd("interaction", "chest", "left"), enemy_adjacent=True,
               enemy_direction="down", entities=[ent("loot", "chest", "left", "very_close")])
assert a == "combat" and r == "override: enemy in reach", (a, r)

# Commanded agent refuses -> fallback chain.
c = PROFILE.initial_counters()
a, r, _ = run(c, command=cmd("interaction", "chest"))
assert a == "navigator" and r == "fallback from interaction", (a, r)
c = PROFILE.initial_counters()
a, r, mv = run(c, command=cmd("combat", "zombie", "up"))
assert a == "combat" and ("ranged", "up") in mv, (a, mv)    # command direction used

# Stalled approach sidesteps, then gives up.
c = PROFILE.initial_counters()
far_chest = dict(command=cmd("interaction", "chest", "left"),
                 entities=[ent("loot", "chest", "left", "close")])
assert run(c, MOVING, **far_chest)[2] == [("move", "left")]
s1 = run(c, STILL, **far_chest)[2]
s2 = run(c, STILL, **far_chest)[2]
assert s1[0][1] != "left" and s2[0][1] != "left" and s1 != s2, (s1, s2)
a, r, _ = run(c, STILL, **far_chest)
assert a != "interaction" and c["interaction_give_ups"] == 1, (a, c["interaction_give_ups"])

# A visible prompt beats the cooldown.
c["_interact_cooldown"] = 3
a, _, mv = run(c, command=cmd("interaction", "prompt"), interact_prompt_visible=True)
assert a == "interaction" and mv[0][0] == "interact", (a, mv)

# Portals are skipped unless the Observer names one.
c = PROFILE.initial_counters()
portal = [ent("portal", "teleport pad", "up", "very_close")]
assert run(c, command=cmd("navigator"), entities=portal)[0] == "navigator"
assert run(c, command=cmd("interaction", "teleport"), entities=portal)[2] == [("interact", "up")]
print("observer routing: OK")


# --- report ---------------------------------------------------------------
def cyc(n, **kw):
    base = dict(cycle=n, scene_state="in_mission", moves=[], entities=[], objective="",
                map_shift=0, delta=1, enemy_count=0)
    return {**base, **kw}


press_villager = [dict(macro="interact", direction="up", purpose="Interact with villager 'farmer'.")]
press_chest = [dict(macro="interact", direction="up", purpose="Interact with loot 'chest'.")]
payload = {"cycles": [
    cyc(1, objective="Reach the Farmer's Fields (marker up)"),
    cyc(2, objective="THE ALLBGERS FROM THE RIFT - Reach the farmer's fields"),
    cyc(3, objective="Defeat the soul corrupted monsters",
        entities=[ent("villager", "farmer", "up", "very_close")], moves=press_villager,
        interact_prompt_visible=True),
    cyc(4, objective="Defeat the soul corrupted monsters"),
    cyc(5, objective="Defeat the soul corrupted monsters",
        entities=[ent("loot", "chest", "up", "very_close")], moves=press_chest),
    cyc(6, objective="Defeat the soul corrupted monsters",
        entities=[ent("loot", "chest", "up", "very_close")]),
]}
r = rep.build(payload)
st = {s["scenario"]: s["status"] for s in r["scenarios"]}
assert st["Saving villagers"] == rep.NOT_ENCOUNTERED, st
assert st["Interacting with villagers"] == rep.PASS, st
assert st["Looting chests"] == rep.FAIL, st        # chest still there afterwards
assert st["Objective reached"] == rep.PASS, st
stages = rep.objective_stages([c for c in payload["cycles"]])
assert len(stages) == 2, stages                    # ALLBGERS misread is not a new stage
assert r["verdict"] == rep.FAIL, r["verdict"]
assert r["run"]["objective_reached"] is True and r["run"]["mission_complete"] is False
# Without the failed chest, unseen scenarios don't block an overall PASS.
payload["cycles"] = payload["cycles"][:4]
r2 = rep.build(payload)
assert r2["verdict"] in (rep.PASS, "INCOMPLETE"), r2["verdict"]
assert all(s["status"] != rep.FAIL for s in r2["scenarios"]), r2["scenarios"]
print("report rules    : OK", r["verdict"], r2["verdict"])
print("OK")
