"""Offline check: combat perception fixes from run dungeons-20261012-143117.

Self-sightings dropped, damage = proof of attack, engagement confirm/hold,
Xbox toast skips the health read, and the hero ring on the model's frame.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tools.game_profiles.dungeons_profile as dp
from tools.game_profiles.dungeons_agents import PROFILE
from tools.game_profiles.dungeons_profile import DungeonsFrameDecision

MOVING = {"bearing": "up_right", "shift": 30.0, "occluded": False}
Q = dict(scene_state="in_mission", player_visible=True, reasoning="t")


def enemy(x, y, kind="illager"):
    return dict(category="enemy", kind=kind, x=x, y=y)


def cycle(c, hp=100, toast=False, **kw):
    """One decision through annotate-equivalent geometry + the supervisor."""
    d = DungeonsFrameDecision(**{**Q, "enemies": "?", **kw})
    d.health_percent = hp
    d._nav = dict(MOVING, toast=toast)
    dp._apply_enemy_geometry(d)
    moves = PROFILE.forced_moves(d, c) or []
    for m in moves:
        PROFILE.on_move_dispatched(m, c)
    return d, d._agent, [(m.action, m.direction) for m in moves]


# 1. Self-sighting: the "illager" at the hero position (cycles 27/29) is the
#    red-armoured hero. It must be dropped - no shot, navigator walks on.
c = PROFILE.initial_counters()
for _ in range(3):
    d, a, mv = cycle(c, enemies="illager right", enemy_direction="right",
                     entities=[enemy(0.51, 0.47)], command=dict(to="combat", direction="right"))
    assert a == "navigator" and not any(m[0] in ("ranged", "fight") for m in mv), (a, mv)
assert c["self_sightings"] == 3 and c["combat_commands_dropped"] == 3, c
print("self-sighting  : OK", mv)

# 2. Geometry: adjacency/direction come from positions, not the model's claim.
d = DungeonsFrameDecision(**Q, enemies="x", enemy_adjacent=False, enemy_direction="up",
                          entities=[enemy(0.58, 0.48), enemy(0.1, 0.1)])
d._nav = {}
dp._apply_enemy_geometry(d)
assert d.enemy_adjacent and d.enemy_direction == "right", (d.enemy_adjacent, d.enemy_direction)
assert d.enemy_count == 1, d.enemy_count          # the far one is not "near"
far = [e for e in d.entities if e.x == 0.1][0]
assert far.distance_estimate == "far" and far.direction == "up_left", far
print("geometry       : OK")

# 3. Confirmation: one far sighting does not trigger a volley; two do.
c = PROFILE.initial_counters()
# (0.72, 0.1): 0.39 right x 16/9 vs 0.38 up -> a true 45-degree up_right.
_, a1, mv1 = cycle(c, entities=[enemy(0.72, 0.1)])
assert a1 == "navigator" and c["unconfirmed_sightings"] == 1, (a1, mv1)
_, a2, mv2 = cycle(c, entities=[enemy(0.72, 0.1)])
assert a2 == "combat" and ("ranged", "up_right") in mv2, (a2, mv2)
print("confirm        : OK", mv2)

# 4. Hold: a missed read mid-fight keeps attacking the last direction for
#    _ENGAGE_HOLD cycles, then releases back to navigation.
held = [cycle(c, enemies="none")[1:] for _ in range(dp._ENGAGE_HOLD + 1)]
assert all(a == "combat" for a, _ in held[:dp._ENGAGE_HOLD]), held
assert held[-1][0] == "navigator", held
assert c["held_engagements"] == dp._ENGAGE_HOLD, c
print("hold           : OK", [a for a, _ in held])

# 5. Damage = proof (cycle 14): health falls, model sees nobody.
#    a) last enemy known -> attack toward it.
c = PROFILE.initial_counters()
cycle(c, hp=95, entities=[enemy(0.2, 0.45)])          # seen once, left
for _ in range(dp._ENGAGE_HOLD + 1):
    cycle(c, hp=95, enemies="none")                   # engagement expires
d, a, mv = cycle(c, hp=80, enemies="none")
assert d._nav["blind_attacker"] == "left" and a == "combat", (a, mv, d._nav)
assert any(m[1] == "left" for m in mv), mv
assert c["blind_hits"] == 1, c
#    a2) a sighting older than _BLIND_MEMORY is too stale to aim at.
c = PROFILE.initial_counters()
cycle(c, hp=95, entities=[enemy(0.2, 0.45)])
for _ in range(dp._BLIND_MEMORY + 1):
    cycle(c, hp=95, enemies="none")
d, _, mv = cycle(c, hp=80, enemies="none")
assert d._nav["blind_attacker"] == "unknown" and mv[0][0] == "dodge", (mv, d._nav)
#    b) never seen -> roll out of the line of fire, and warn the model.
#       (a hit that stays above the potion threshold; a big one drinks first)
c = PROFILE.initial_counters()
cycle(c, hp=100, enemies="none")
d, a, mv = cycle(c, hp=88, enemies="none")
assert d._nav["blind_attacker"] == "unknown" and mv[0][0] == "dodge", (a, mv)
assert "no attacker was found" in c["_hint"], c["_hint"]
assert "WARNING" in PROFILE.extra_session_text(c)
print("damage-as-proof: OK", mv)

# 6. Navigator commanded but a confirmed adjacent enemy is there -> combat.
c = PROFILE.initial_counters()
_, a, mv = cycle(c, entities=[enemy(0.56, 0.5)], command=dict(to="navigator"))
assert a == "combat" and mv[-1][0] == "fight", (a, mv)
print("nav->combat    : OK", mv)

# 7. Toast (cycle 39): health read skipped, no potion, baseline kept.
frame = np.zeros((1080, 1920, 3), np.uint8)
frame[910:1000, 660:1270] = (40, 160, 20)             # BGR saturated green bar
assert dp._toast_visible(frame) and not dp._toast_visible(np.zeros_like(frame))
d = DungeonsFrameDecision(**Q, enemies="none", health_percent=100)
dp._annotate(d, frame)
assert d._nav.get("toast") and d._nav.get("hud"), d._nav
c = PROFILE.initial_counters()
c["_last_health"] = 100
d.health_percent = 23
assert dp._potion_check(d, c) is None and c["_last_health"] == 100, c
print("toast          : OK")

# 8. Model frame gets the hero ring; the raw frame is untouched.
raw = np.zeros((1080, 1920, 3), np.uint8)
seen = PROFILE.prepare_vision_frame(raw)
assert raw.sum() == 0 and seen.sum() > 0
assert tuple(seen[int(1080 * 0.48), 960 + int(1080 * dp._HERO_RING_R)]) != (0, 0, 0)
print("hero ring      : OK")

# 9. No positions given: model fields are used unchanged (backward compatible).
d = DungeonsFrameDecision(**Q, enemies="zombie", enemy_adjacent=True, enemy_direction="left",
                          entities=[dict(category="enemy", kind="zombie", direction="left")])
d._nav = {}
dp._apply_enemy_geometry(d)
assert d.enemy_adjacent and d.enemy_direction == "left" and d._nav["enemy_geo"] is None
print("no positions   : OK")

# 10. The real toast frame from the failed run, when the artifact is present.
run = sorted((ROOT / "artifacts" / "runs").glob("dungeons-20261012-143117"))
if run:
    import cv2
    frames = run[0] / "frames"
    toast = [int(p.stem.split("-")[1]) for p in sorted(frames.glob("dungeons-*-before.jpg"))
             if dp._toast_visible(cv2.imread(str(p)))]
    assert toast == [39], toast
    print("real frames    : OK toast only on cycle", toast)
print("OK")
