"""Offline check: every Dungeons action resolves to real GIMX controls (fake pad)."""
import sys
from pathlib import Path
from types import SimpleNamespace

import yaml

ROOT = Path(__file__).resolve().parent.parent
for sub in ("core", "tools", "agents", "graph", "tools/game_profiles"):
    sys.path.insert(0, str(ROOT / sub))

import game_profiles.dungeons_profile as dp
from game_profiles.dungeons_profile import DungeonsFrameDecision, DungeonsMove, PROFILE

controls = yaml.safe_load((ROOT.parent / "Xbox-Automation-Python" / "config" / "controls.yaml")
                          .read_text(encoding="utf-8"))


class FakePad:
    def __init__(self):
        self.cfg = SimpleNamespace(buttons=controls["buttons"], triggers=controls["triggers"],
                                   sticks=controls["sticks"])
        self.sent = []

    def _send_events(self, events, label):
        self.sent.append((label, events))
        return True

    def press(self, name, duration=0.2):
        spec = self.cfg.buttons[name]
        self.sent.append((f"press:{name}", [(spec["gimx"], 1)]))
        return True


pad = FakePad()
ctx = SimpleNamespace(hardware=SimpleNamespace(pad=lambda: pad))

for key, entry in dp.all_bindings().items():
    table = pad.cfg.triggers if entry["kind"] == "trigger" else pad.cfg.buttons
    assert entry["button"] in table, f"{key}: '{entry['button']}' not a {entry['kind']}"

for action in ["move", "melee", "ranged", "dodge", "forward_dodge", "interact",
               "artifact_1", "artifact_2", "artifact_3", "advance_prompt", "potion"]:
    pad.sent.clear()
    res = dp.execute_move(ctx, DungeonsMove(action=action, direction="up_right", duration=0.3))
    assert res["dispatched"], (action, res)
    print(f"{action:15s} -> {[(l, e) for l, e in pad.sent if not l.endswith('release')]}")

pad.sent.clear()
r = dp.execute_moves(ctx, [DungeonsMove(action="move", direction="left"),
                           DungeonsMove(action="forward_dodge")])
print("move+forward_dodge merged ->", pad.sent[0])
pad.sent.clear()
dp.execute_moves(ctx, [DungeonsMove(action="move", direction="left"),
                       DungeonsMove(action="melee", direction="up")])
print("move+melee (stick clash) -> sequential calls:", len(pad.sent))

base = dict(scene_state="in_mission", player_visible=True, enemies="zombie up",
            enemy_adjacent=True, enemy_direction="up", reasoning="t")
c = PROFILE.initial_counters()
print("reflex melee   :", PROFILE.forced_moves(DungeonsFrameDecision(**base), c))
print("reflex low hp  :", PROFILE.forced_moves(DungeonsFrameDecision(**base, health_low=True), c))

c = PROFILE.initial_counters()
crowd = DungeonsFrameDecision(**base, enemy_count=3)
runs = [[m.action for m in PROFILE.forced_moves(crowd, c)] for _ in range(3)]
print("crowd x3       :", runs, "retreats=", c.get("retreats"))
assert "dodge" in runs[0] and "move" in runs[0] and "fight" not in runs[0], runs[0]
assert "dodge" in runs[1] and "fight" not in runs[1], runs[1]
assert "fight" in runs[2], runs[2]
two = [m.action for m in PROFILE.forced_moves(DungeonsFrameDecision(**base, enemy_count=2), c)]
assert "fight" in two, two

# Health monitor
c = PROFILE.initial_counters()
quiet = dict(scene_state="in_mission", player_visible=True, enemies="none", reasoning="t")
assert PROFILE.forced_moves(DungeonsFrameDecision(**quiet, health_percent=100), c) is None
low = [m.action for m in PROFILE.forced_moves(DungeonsFrameDecision(**quiet, health_percent=40), c)]
assert low[0] == "potion", low
again = PROFILE.forced_moves(DungeonsFrameDecision(**quiet, health_percent=35), c)
assert again is None, again            # cooldown
c = PROFILE.initial_counters()
PROFILE.forced_moves(DungeonsFrameDecision(**base, health_percent=100), c)
hit = [m.action for m in PROFILE.forced_moves(DungeonsFrameDecision(**base, health_percent=70), c)]
assert hit[:2] == ["potion", "dodge"], hit   # sharp drop while adjacent
c = PROFILE.initial_counters()
nr = PROFILE.forced_moves(DungeonsFrameDecision(**base, health_percent=30, potion_ready=False), c)
assert all(m.action != "potion" for m in nr), nr
print("potion monitor : OK", low, hit)

# --- potion regressions -----------------------------------------------------
# hp == 0 must still refresh the baseline, else the post-revive reading looks
# like a massive drop and wastes a potion.
c = PROFILE.initial_counters()
PROFILE.forced_moves(DungeonsFrameDecision(**quiet, health_percent=90), c)
PROFILE.forced_moves(DungeonsFrameDecision(**quiet, health_percent=0), c)
assert c["_last_health"] == 90, c["_last_health"]
assert dp._POTION_COOLDOWN_S == 30.0, dp._POTION_COOLDOWN_S

# An unconfirmed drink (no heart refill) retries instead of locking out 30s.
c = PROFILE.initial_counters()
first = PROFILE.forced_moves(DungeonsFrameDecision(**quiet, health_percent=40), c)
assert first[0].action == "potion", first
assert c["_potion_pending"] is True
c["_last_potion"] -= dp._POTION_RETRY_S + 0.1          # retry window elapsed
retry = PROFILE.forced_moves(DungeonsFrameDecision(**quiet, health_percent=40), c)
assert retry and retry[0].action == "potion", retry     # still low -> retry
# Now the heart gauge refills: the drink is confirmed and the flag clears.
PROFILE.forced_moves(DungeonsFrameDecision(**quiet, health_percent=80), c)
assert c["_potion_pending"] is False, c
assert c["potions_confirmed"] == 1, c
print("potion confirm : OK")

# --- yield hatch ------------------------------------------------------------
# Enemy visible but NOT adjacent + an interact prompt -> the model's own plan
# must survive. This is the bug that starved every non-combat behaviour.
c = PROFILE.initial_counters()
ranged_enemy = dict(scene_state="in_mission", player_visible=True,
                    enemies="zombie up", enemy_adjacent=False,
                    enemy_direction="up", reasoning="t")
plan = [DungeonsMove(action="interact", direction="right", purpose="open chest")]
yielded = PROFILE.forced_moves(
    DungeonsFrameDecision(**ranged_enemy, interact_prompt_visible=True, moves=plan), c)
assert yielded is None, yielded
assert c["model_yields"] == 1, c
# Without the prompt the reflex still shoots.
shoots = PROFILE.forced_moves(DungeonsFrameDecision(**ranged_enemy, moves=plan), c)
assert any(m.action == "ranged" for m in shoots), shoots
# An ADJACENT enemy outranks the prompt: survival first.
fights = PROFILE.forced_moves(
    DungeonsFrameDecision(**base, interact_prompt_visible=True, moves=plan), c)
assert any(m.action == "fight" for m in fights), fights
print("yield hatch    : OK")

# --- rescue priority --------------------------------------------------------
# Two enemies: the nearest to the hero is 'down', but the one mauling the
# villager is 'left'. The bot must swing at the villager's attacker.
c = PROFILE.initial_counters()
ents = [
    dict(category="enemy", kind="zombie", direction="down",
         distance_estimate="close", is_threat=True),
    dict(category="enemy", kind="zombie", direction="left",
         distance_estimate="close", is_threat=True, threatens_villager=True),
    dict(category="villager", kind="caged villager", direction="left",
         distance_estimate="close"),
]
res = PROFILE.forced_moves(DungeonsFrameDecision(
    scene_state="in_mission", player_visible=True, enemies="two zombies",
    enemy_adjacent=False, enemy_direction="down", objective_kind="rescue",
    villager_in_danger=True, entities=ents, reasoning="t"), c)
assert res[0].action == "fight" and res[0].direction == "left", res
assert c["rescue_attacks"] == 1, c

# No attacker left: walk to the villager, then free them when adjacent.
c = PROFILE.initial_counters()
far_v = [dict(category="villager", kind="caged villager", direction="up_right",
              distance_estimate="far")]
walk = PROFILE.forced_moves(DungeonsFrameDecision(
    scene_state="in_mission", player_visible=True, enemies="none",
    objective_kind="rescue", villager_in_danger=True, entities=far_v,
    reasoning="t"), c)
assert walk[0].action == "move" and walk[0].direction == "up_right", walk
near_v = [dict(category="villager", kind="caged villager", direction="up_right",
               distance_estimate="very_close")]
free = PROFILE.forced_moves(DungeonsFrameDecision(
    scene_state="in_mission", player_visible=True, enemies="none",
    objective_kind="rescue", villager_in_danger=True, entities=near_v,
    reasoning="t"), c)
assert free[0].action == "interact", free
print("rescue priority: OK")

# --- objective bearing ------------------------------------------------------
# Fallback navigation must follow the minimap marker, not stale prose.
fb = PROFILE.fallback_moves(DungeonsFrameDecision(
    **quiet, objective_bearing="down_left", progress_direction="up"))
assert fb[0].direction == "down_left", fb
fb2 = PROFILE.fallback_moves(DungeonsFrameDecision(
    **quiet, progress_direction="right"))
assert fb2[0].direction == "right", fb2
print("objective nav  : OK")

# --- loose bools ------------------------------------------------------------
lb = DungeonsFrameDecision(**{**quiet, "enemy_adjacent": "unknown", "potion_ready": "yes"})
assert lb.enemy_adjacent is False and lb.potion_ready is True, lb
print("loose bools    : OK")

# --- wall escape / minimap steering (synthetic nav readings) -----------------
def nav_cycle(c, nav, model_dir="right"):
    d = DungeonsFrameDecision(**quiet, moves=[DungeonsMove(action="move", direction=model_dir, duration=3.0)])
    d._nav = nav
    mv = PROFILE.forced_moves(d, c) or d.moves
    for m in mv:
        PROFILE.on_move_dispatched(m, c)
    return [(m.action, m.direction) for m in mv]

still = {"bearing": "up_right", "shift": 0.3, "occluded": False}
c = PROFILE.initial_counters()
seq = [nav_cycle(c, still) for _ in range(4)]
assert seq[0] == [("move", "right")] and seq[1] == [("move", "right")], seq
assert seq[2][0][0] == "dodge" and seq[2][0][1] != "right", seq     # 2 still cycles -> escape
assert seq[3][0][0] == "move" and seq[3][0][1] != "right", seq      # detour follow-up
# Wall glow escapes after a single still cycle.
c = PROFILE.initial_counters()
glow = {**still, "occluded": True}
g = [nav_cycle(c, glow) for _ in range(2)]
assert g[1][0][0] == "dodge", g
# Moving freely: model's heading kept; heading opposite the map marker is corrected.
c = PROFILE.initial_counters()
moving = {"bearing": "up_right", "shift": 30.0, "occluded": False}
assert nav_cycle(c, moving) == [("move", "right")]
assert nav_cycle(c, moving, model_dir="down_left") == [("move", "up_right")]
assert c["map_steers"] == 1 and c["wall_escapes"] == 0, c
print("wall escape    : OK", seq[2], g[1])

print("OK")
