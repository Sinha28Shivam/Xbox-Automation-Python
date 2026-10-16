"""Dungeons combat speed: sustain while thinking, compact schema, tuning.

No hardware: a fake pad (records GIMX events), a synthetic HUD frame whose
heart gauge fill sets the pixel health, and a fake vision decider that sleeps
like the real model (the sustain must fill exactly that gap).
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import yaml

from tools import gameplay_engine as ge
from tools.game_profiles import dungeons_profile as dp
from tools.game_profiles import dungeons_speed as sp
from tools.game_profiles.dungeons_agents import PROFILE as AGENTS_PROFILE

ROOT = Path(__file__).resolve().parents[1]
_CONTROLS = yaml.safe_load((ROOT.parent / "Xbox-Automation-Python" / "config" / "controls.yaml")
                           .read_text(encoding="utf-8"))


class FakePad:
    def __init__(self) -> None:
        self.cfg = SimpleNamespace(buttons=_CONTROLS["buttons"], triggers=_CONTROLS["triggers"],
                                   sticks=_CONTROLS["sticks"])
        self.sent: list[str] = []
        self.lock = threading.Lock()

    def _send_events(self, events: Any, label: str) -> bool:
        with self.lock:
            self.sent.append(label)
        return True

    def press(self, name: str, duration: float = 0.2) -> bool:
        with self.lock:
            self.sent.append(f"press:{name}")
        return True

    def count(self, prefix: str) -> int:
        with self.lock:
            return sum(1 for s in self.sent if s.startswith(prefix))


def hud_frame(health: int) -> np.ndarray:
    """1920x1080 grey frame (not 'blank' to the engine) with the blue artifact
    hex (HUD visible) and the heart gauge filled to `health` percent."""
    f = np.full((1080, 1920, 3), 40, dtype=np.uint8)
    x0, y0, x1, y1 = (int(1920 * dp._HUD_HEX_BOX[0]), int(1080 * dp._HUD_HEX_BOX[1]),
                      int(1920 * dp._HUD_HEX_BOX[2]), int(1080 * dp._HUD_HEX_BOX[3]))
    f[y0:y1, x0:x1] = (230, 140, 40)
    hy0, hy1 = int(1080 * dp._HEART_Y[0]), int(1080 * dp._HEART_Y[1])
    hx0, hx1 = int(1920 * dp._HEART_X[0]), int(1920 * dp._HEART_X[1])
    rows = round(dp._HEART_FULL_ROWS * (hy1 - hy0) / 140 * health / 100)
    if rows:
        f[hy1 - rows:hy1, hx0:hx1] = (20, 20, 220)
    return f


@pytest.fixture(autouse=True)
def _legacy_free_tuning():
    saved = dict(dp.TUNING)
    yield
    dp.TUNING.clear()
    dp.TUNING.update(saved)


def _ctx(pad: FakePad) -> Any:
    return SimpleNamespace(hardware=SimpleNamespace(pad=lambda: pad))


def _engaged_snap(**kw: Any) -> dict[str, Any]:
    return {**dp._initial_counters(), "_engage_left": 3,
            "_sustain_action": ("ranged", "right"), "_last_health": 100, **kw}


def test_pixel_health_fixture_matches_reader():
    assert abs(dp._pixel_health(hud_frame(100)) - 100) <= 2
    assert abs(dp._pixel_health(hud_frame(50)) - 50) <= 3
    assert dp._hud_visible(hud_frame(80))


# --- A. sustain --------------------------------------------------------------
def test_sustain_keeps_attacking_until_stopped():
    pad, stop = FakePad(), threading.Event()
    frame = hud_frame(100)
    threading.Timer(1.6, stop.set).start()
    r = sp.sustain(_ctx(pad), _engaged_snap(), stop, lambda: frame, 12.0)
    assert r["stopped_by"] == "next decision"
    assert r["volleys"] >= 3 and 1.4 <= r["seconds"] <= 2.4, r
    assert pad.count("ranged:press") == r["volleys"]


def test_sustain_respects_cap():
    pad = FakePad()
    frame = hud_frame(100)
    r = sp.sustain(_ctx(pad), _engaged_snap(_sustain_action=("fight", "left")),
                   threading.Event(), lambda: frame, 1.0)
    assert r["stopped_by"].startswith("cap") and r["seconds"] < 1.8
    assert r["swings"] >= 2 and pad.count("fight_dodge") == 0  # no rolls mid-sustain


def test_sustain_breaks_off_to_drink_on_health_drop():
    pad = FakePad()
    frames = iter([hud_frame(100), hud_frame(60)])
    r = sp.sustain(_ctx(pad), _engaged_snap(), threading.Event(),
                   lambda: next(frames, hud_frame(60)), 12.0)
    assert "drank potion" in r["stopped_by"] and r["danger_hp"] <= 62
    assert pad.count("potion:press") == 1 and pad.count("dodge:press") == 1
    c = dp._initial_counters()
    sp.merge_sustain(c, r)
    assert c["potions"] == 1 and c["_potion_pending"] and c["sustain_danger_stops"] == 1


def test_sustain_uses_learned_potion_threshold():
    # 78% is below the learned 80% (G001) but above the default 70%.
    frame = hud_frame(78)
    stop = threading.Event()
    threading.Timer(0.6, stop.set).start()
    plain = sp.sustain(_ctx(FakePad()), _engaged_snap(_last_health=82), stop,
                       lambda: frame, 12.0)
    assert "danger_hp" not in plain
    learned = sp.sustain(_ctx(FakePad()),
                         _engaged_snap(_last_health=82, _learn_potion_threshold=80),
                         threading.Event(), lambda: frame, 12.0)
    assert "drank potion" in learned["stopped_by"]


def test_sustain_does_nothing_out_of_combat():
    pad = FakePad()
    frame = hud_frame(100)
    for snap in (_engaged_snap(_engage_left=0),
                 _engaged_snap(_sustain_action=("move", "up")),
                 _engaged_snap(_sustain_action=None)):
        r = sp.sustain(_ctx(pad), snap, threading.Event(), lambda: frame, 12.0)
        assert r["slices"] == 0 and r["stopped_by"] == "not engaged"
    assert pad.sent == []


def test_sustain_stops_when_hud_disappears():
    r = sp.sustain(_ctx(FakePad()), _engaged_snap(), threading.Event(),
                   lambda: np.zeros((1080, 1920, 3), dtype=np.uint8), 12.0)
    assert r["stopped_by"].startswith("HUD gone") and r["slices"] == 0


def test_dispatch_records_what_to_sustain():
    c = dp._initial_counters()
    dp._on_move_dispatched(dp.DungeonsMove(action="artifact_1"), c)
    dp._on_move_dispatched(dp.DungeonsMove(action="ranged", direction="right"), c)
    assert c["_sustain_action"] == ("ranged", "right")
    dp._on_move_dispatched(dp.DungeonsMove(action="move", direction="up"), c)
    assert c["_sustain_action"] is None


# --- B/C. compact schema and frame size --------------------------------------
def test_fast_decision_converts_and_routes_to_combat():
    fast = sp.DungeonsCombatDecision(
        scene_state="in_mission", enemies="pillager right",
        enemy_list=[sp.CombatEnemy(kind="pillager", x=0.8, y=0.48)], health_percent=90)
    d = sp.to_full_decision(fast)
    d._nav = {"bearing": "up", "shift": 30.0, "occluded": False}
    dp._apply_enemy_geometry(d)
    assert d.enemy_direction == "right"
    c = {**AGENTS_PROFILE.initial_counters(), "_engage_left": 3}
    moves = AGENTS_PROFILE.forced_moves(d, c)
    assert d._agent == "combat" and moves[-1].action == "ranged"


def test_full_refresh_cadence_and_size():
    use_fast = sp.make_use_fast(3)
    c = {"_engage_left": 3}
    picks = [use_fast(c, "in_mission") for _ in range(6)]
    assert picks == [True, True, False, True, True, False]
    assert use_fast({"_engage_left": 0}, "in_mission") is False
    assert use_fast({"_engage_left": 3}, "death") is False
    size = sp.make_vision_size(640, 65)
    assert size({"_engage_left": 2}) == (640, 65) and size({"_engage_left": 0}) is None


def test_encode_frame_width_override():
    import base64

    import cv2
    f = hud_frame(100)
    small = cv2.imdecode(np.frombuffer(base64.b64decode(ge.encode_frame(f, 640, 65)),
                                       np.uint8), cv2.IMREAD_COLOR)
    full = cv2.imdecode(np.frombuffer(base64.b64decode(ge.encode_frame(f)), np.uint8),
                        cv2.IMREAD_COLOR)
    assert small.shape[1] == 640 and full.shape[1] == ge.VISION_WIDTH


# --- D. tuning ----------------------------------------------------------------
def test_tuning_from_settings_and_legacy():
    s = SimpleNamespace(get=lambda k, d=None: {"gameplay.dungeons.fight_swings": "5",
                                               "gameplay.dungeons.sustain_combat": "false"}
                        .get(k, d))
    cfg = sp.load_speed(s)
    assert cfg["fight_swings"] == 5 and cfg["sustain_combat"] is False
    sp.apply_tuning(cfg)
    pad = FakePad()
    r = dp._fight(pad, "left")
    assert r["swings"] == 5 and pad.count("fight:press") == 5
    dp.TUNING.update(dp.LEGACY_TUNING)
    pad = FakePad()
    r = dp._fight(pad, "left")
    assert r["swings"] == 4 and r["sidestep"] and pad.count("fight_dodge:press") == 1


def test_artifact_time_cooldown():
    dp.TUNING["artifact_cooldown_s"] = 8.0
    c = dp._initial_counters()
    c["combat_cycles"] = 2
    assert dp._artifact_due(c) and not dp._artifact_due(c)
    c["_last_artifact_t"] = time.monotonic() - 9
    assert dp._artifact_due(c)
    dp.TUNING["artifact_cooldown_s"] = 0.0
    c["combat_cycles"] = 4
    assert dp._artifact_due(c)  # legacy: every 3rd combat cycle


def test_flags_off_profile_is_unchanged():
    cfg = {**sp.DEFAULTS, "sustain_combat": False, "fast_combat_schema": False,
           "combat_vision_width": 0}
    p = sp.with_speed(AGENTS_PROFILE, cfg)
    assert p.sustain is None and p.fast_frame_model is None
    assert p.vision_size({"_engage_left": 3}) is None


# --- engine: sustain fills the model's think time ----------------------------
class _Camera:
    def __init__(self) -> None:
        self.frame = hud_frame(100)

    def grab(self, allow_blank: bool = True) -> Any:
        return self.frame.copy()


class _Decider:
    """Answers like the model: enemy to the right, after a real delay."""
    def __init__(self, delay: float) -> None:
        self.delay, self.calls = delay, []

    def invoke(self, messages: Any) -> Any:
        time.sleep(self.delay)
        text = messages[0].content[0]["text"]
        fast = text.startswith(sp.COMBAT_PROMPT[:40])
        self.calls.append("fast" if fast else "full")
        enemy = {"kind": "pillager", "x": 0.62, "y": 0.48}  # close, not adjacent
        if fast:
            return sp.DungeonsCombatDecision(scene_state="in_mission",
                                             enemies="pillager", enemy_list=[enemy])
        return dp.DungeonsFrameDecision(
            scene_state="in_mission", player_visible=True, enemies="pillager",
            reasoning="t", entities=[{"category": "enemy", "kind": "pillager", **enemy}],
            command={"to": "combat"})


def _run_engine(monkeypatch, profile, cycles: int, delay: float) -> tuple[dict, FakePad, _Decider]:
    pad, cam = FakePad(), _Camera()
    deciders = {}

    def fake_build(ctx, model):
        deciders[model.__name__] = deciders.get("shared") or _Decider(delay)
        deciders["shared"] = deciders[model.__name__]
        return deciders["shared"]
    monkeypatch.setattr(ge, "build_vision_decider", fake_build)
    monkeypatch.setattr(ge, "_current_provider", lambda ctx: None)
    monkeypatch.setattr(ge, "frame_delta", lambda ctx, a, b: 5.0)
    monkeypatch.setattr(dp, "_read_nav", lambda f: {"bearing": "up", "shift": 30.0,
                                                    "occluded": False, "hud": True})
    ctx = SimpleNamespace(
        hardware=SimpleNamespace(pad=lambda: pad, capture=lambda: cam),
        artifacts=SimpleNamespace(enabled=False, save_json=lambda *a, **k: None),
        settings=None, dry_run=False)
    result = ge.run_gameplay_loop(ctx, profile, max_cycles=cycles, cycle_delay=0.0,
                                  settle_after_move=0.0, pipeline=True, overlap_act=True,
                                  overlap_lead=0.05)
    return result, pad, deciders["shared"]


def test_engine_sustain_attacks_during_think(monkeypatch):
    dp.TUNING.update(dp.LEGACY_TUNING)
    # 2.5 s think (the real model: ~5.6 s); legacy 0.7 s volley leaves ~1.8 s idle.
    base, base_pad, _ = _run_engine(monkeypatch, AGENTS_PROFILE, cycles=5, delay=2.5)
    fast_profile = sp.with_speed(AGENTS_PROFILE, {**sp.DEFAULTS, "fast_combat_schema": False,
                                                  "combat_vision_width": 0})
    sped, sped_pad, _ = _run_engine(monkeypatch, fast_profile, cycles=5, delay=2.5)
    old = base_pad.count("ranged:press")
    new = sped_pad.count("ranged:press")
    assert new >= 2 * old, (old, new)            # volleys while the model thought
    assert sped["sustain_slices"] >= 4 and sped["sustain_longest_s"] <= 12
    assert any(c.get("sustain", {}).get("slices") for c in sped["cycles"])


def test_engine_uses_fast_schema_and_refreshes(monkeypatch):
    profile = sp.with_speed(AGENTS_PROFILE, {**sp.DEFAULTS, "sustain_combat": False})
    result, _, decider = _run_engine(monkeypatch, profile, cycles=7, delay=0.01)
    modes = [c["decide_mode"] for c in result["cycles"]]
    assert modes[0] == "full" and "fast" in modes
    assert decider.calls.count("full") >= 2      # periodic full refresh
    assert all(c["agent"] == "combat" for c in result["cycles"][1:])
