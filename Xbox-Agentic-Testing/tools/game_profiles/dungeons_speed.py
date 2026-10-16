"""Combat speed for Minecraft Dungeons (settings gameplay.dungeons).

Measured on dungeons-20261016-114012: the model takes ~5.6 s per combat
decision, but each combat batch (one 0.7 s volley or one melee burst) ends in
~1 s, so the hero stood idle ~4 s of every ~5.5 s in a fight. Four fixes,
each behind its own flag (all on by default):

  sustain_combat      keep attacking on the input thread while the model
                      thinks (gameplay_engine GameProfile.sustain). Checks the
                      heart gauge between short slices and breaks off to drink
                      or roll at once; stops when the next decision is ready,
                      the HUD disappears, or after sustain_max_s.
  fast_combat_schema  while engaged, ask for a 9-field combat reading with a
                      short prompt instead of the full ~30-field schema; a full
                      decision every full_refresh_every cycles keeps objective,
                      loot and villager reading alive.
  combat_vision_width smaller frame to the model while engaged (0 = off).
  tuning              longer volley, longer melee burst, roll every 2nd burst,
                      artifacts on a time cooldown (dungeons_profile.TUNING).
"""
from __future__ import annotations

import threading
import time
from dataclasses import replace
from typing import Any, Literal

from pydantic import Field

from tools.gameplay_engine import GameProfile
from tools.registry import ToolContext

from . import dungeons_profile as dp
from .dungeons_profile import DungeonsFrameDecision, DungeonsMove, _LooseBools

DEFAULTS: dict[str, Any] = {
    "sustain_combat": True,
    "sustain_max_s": 12.0,
    "fast_combat_schema": True,
    "full_refresh_every": 3,
    "combat_vision_width": 640,
    "combat_jpeg_quality": 65,
    **dp.TUNING,
}
_TUNING_KEYS = tuple(dp.TUNING)

_SLICE_SWINGS = 2          # melee swings per sustain slice (~0.4 s)
_SLICE_RANGED_S = 0.45     # crossbow hold per sustain slice
_SUSTAIN_ACTIONS = ("fight", "ranged")

_HEADING = Literal["up", "down", "left", "right",
                   "up_left", "up_right", "down_left", "down_right", "none"]


# --- settings ----------------------------------------------------------------
def _flag(value: Any) -> bool:
    return str(value).strip().lower() not in ("false", "0", "no", "off", "")


def load_speed(settings: Any) -> dict[str, Any]:
    """gameplay.dungeons.* merged over DEFAULTS, types coerced."""
    cfg = dict(DEFAULTS)
    get = getattr(settings, "get", None)
    for key, default in DEFAULTS.items():
        raw = get(f"gameplay.dungeons.{key}", default) if callable(get) else default
        try:
            cfg[key] = (_flag(raw) if isinstance(default, bool)
                        else type(default)(raw))
        except (TypeError, ValueError):
            cfg[key] = default
    return cfg


def apply_tuning(cfg: dict[str, Any]) -> None:
    """Combat timing knobs read by dungeons_profile (process-wide)."""
    dp.TUNING.update({k: cfg[k] for k in _TUNING_KEYS if k in cfg})


# --- B. compact combat decision ---------------------------------------------
class CombatEnemy(_LooseBools):
    kind: str = Field(default="enemy", description="e.g. 'pillager', 'zombie'.")
    x: float | None = Field(default=None, ge=0.0, le=1.0,
                            description="Horizontal centre 0 (left) - 1 (right).")
    y: float | None = Field(default=None, ge=0.0, le=1.0,
                            description="Vertical centre 0 (top) - 1 (bottom).")


class DungeonsCombatDecision(_LooseBools):
    """The few fields a fight needs; everything else keeps its default."""
    scene_state: Literal["in_mission", "menu_or_prompt", "loading_or_cutscene",
                         "death", "mission_complete"] = Field(
        description="Classification of the current screen.")
    enemies: str = Field(default="none", description="Visible enemies, or 'none'.")
    enemy_list: list[CombatEnemy] = Field(
        default_factory=list, description="Every visible enemy with its x/y.")
    enemy_adjacent: bool = Field(default=False)
    enemy_count: int = Field(default=0, ge=0)
    enemy_direction: _HEADING = Field(default="none")
    health_percent: int = Field(default=100, ge=0, le=100)
    potion_ready: bool = Field(default=True)
    reasoning: str = Field(default="", description="ONE short sentence.")


COMBAT_PROMPT = """\
You are the OBSERVER for Minecraft Dungeons on a real Xbox, IN A FIGHT.
The image is the LIVE top-down screen. The hero is at the centre inside the
yellow ring and is NEVER an enemy. Report only what is visible:
- scene_state first ("YOU ARE DOWNED" = death; loading/menu as such).
- enemy_list: EVERY enemy (illagers, pillagers with crossbows, vindicators,
  zombies, skeletons, spiders, creepers) with x/y = screen position 0-1 of
  its centre. Scan the edges: crossbow pillagers shoot from far away. Red
  tint or arrows at the hero mean an attacker is on screen - find it.
- Only list enemies you can actually see; never guess at dark edges. A fight
  is over when no enemy is near: say enemies 'none' and list none.
- enemy_direction / enemy_adjacent / enemy_count are a fallback only.
- potion_ready from the potion icon. One short sentence of reasoning.
Keep it short: the system attacks while you look.
"""


def to_full_decision(fast: Any) -> DungeonsFrameDecision:
    entities = [{"category": "enemy", "kind": e.kind, "x": e.x, "y": e.y,
                 "is_threat": True} for e in fast.enemy_list]
    return DungeonsFrameDecision(
        scene_state=fast.scene_state, player_visible=True, enemies=fast.enemies,
        enemy_adjacent=fast.enemy_adjacent, enemy_count=fast.enemy_count,
        enemy_direction=fast.enemy_direction, health_percent=fast.health_percent,
        potion_ready=fast.potion_ready, reasoning=fast.reasoning,
        entities=entities,
        command={"to": "combat" if fast.enemy_direction != "none" or entities
                 else "navigator"})


def _engaged(counters: dict[str, Any]) -> bool:
    return int(counters.get("_engage_left", 0) or 0) > 0


def make_use_fast(refresh_every: int) -> Any:
    def use_fast(counters: dict[str, Any], last_scene: str | None) -> bool:
        ok = (last_scene == "in_mission" and _engaged(counters)
              and counters.get("_fast_streak", 0) < max(1, refresh_every) - 1)
        counters["_fast_streak"] = counters.get("_fast_streak", 0) + 1 if ok else 0
        counters["fast_decisions" if ok else "full_decisions"] = \
            counters.get("fast_decisions" if ok else "full_decisions", 0) + 1
        return ok
    return use_fast


# --- C. smaller frame while engaged -----------------------------------------
def make_vision_size(width: int, quality: int) -> Any:
    def vision_size(counters: dict[str, Any]) -> tuple[int, int] | None:
        return (width, quality) if width > 0 and _engaged(counters) else None
    return vision_size


# --- A. sustained combat -----------------------------------------------------
def sustain(ctx: ToolContext, snap: dict[str, Any], stop: threading.Event,
            grab: Any, max_s: float) -> dict[str, Any]:
    """Repeat the last combat action until stop / cap / danger.

    Reads only the snapshot; returns what happened for merge_sustain.
    """
    action = snap.get("_sustain_action")
    out: dict[str, Any] = {"seconds": 0.0, "slices": 0, "swings": 0, "volleys": 0,
                           "stopped_by": "not engaged"}
    if not action or action[0] not in _SUSTAIN_ACTIONS or not _engaged(snap):
        return out
    kind, direction = action
    if direction in (None, "none"):
        return out
    pad = ctx.hardware.pad()
    threshold = int(snap.get("_learn_potion_threshold") or dp._POTION_THRESHOLD)
    drop = int(snap.get("_learn_potion_drop") or dp._POTION_DROP)
    base_hp = int(snap.get("_last_health", 100) or 100)
    start = time.monotonic()
    out["stopped_by"] = "next decision"
    while not stop.is_set():
        if time.monotonic() - start >= max_s:
            out["stopped_by"] = f"cap {max_s:.0f}s"
            break
        frame = grab()
        if frame is not None and not dp._toast_visible(frame):
            if not dp._hud_visible(frame):
                out["stopped_by"] = "HUD gone (downed, loading or menu)"
                break
            hp = dp._pixel_health(frame)
            if hp is not None:
                out["min_health"] = min(out.get("min_health", 100), hp)
                if hp == 0:
                    out["stopped_by"] = "downed"
                    break
                dropped = base_hp - hp >= drop
                # Already low with the potion on cooldown and no new damage:
                # stopping would only idle the hero, so keep fighting.
                if dropped or (hp <= threshold and _potion_cooled(snap)):
                    out.update(_danger(ctx, snap, hp, direction))
                    break
        if stop.is_set():
            break
        if kind == "fight":
            dp._fight(pad, direction, swings=_SLICE_SWINGS, sidestep=False)
            out["swings"] += _SLICE_SWINGS
        else:
            dp.execute_move(ctx, DungeonsMove(action="ranged", direction=direction,
                                              duration=_SLICE_RANGED_S))
            out["volleys"] += 1
        out["slices"] += 1
    out["seconds"] = round(time.monotonic() - start, 2)
    out["action"] = kind
    out["direction"] = direction
    return out


def _potion_cooled(snap: dict[str, Any]) -> bool:
    wait_s = dp._POTION_RETRY_S if snap.get("_potion_pending") else dp._POTION_COOLDOWN_S
    return time.monotonic() - float(snap.get("_last_potion", float("-inf"))) >= wait_s


def _danger(ctx: ToolContext, snap: dict[str, Any], hp: int,
            enemy_dir: str) -> dict[str, Any]:
    """Health fell during a sustain: drink if the potion is ready, then roll away."""
    res: dict[str, Any] = {"stopped_by": f"health {hp}%", "danger_hp": hp}
    if _potion_cooled(snap):
        dp.execute_move(ctx, DungeonsMove(action="potion",
                                          purpose=f"Sustain: health {hp}%."))
        res["potion_t"] = time.monotonic()
        res["stopped_by"] += " - drank potion"
    away = dp._OPPOSITE.get(enemy_dir, "down")
    dp.execute_move(ctx, DungeonsMove(action="dodge", direction=away))
    res["dodged"] = away
    return res


def merge_sustain(counters: dict[str, Any], r: dict[str, Any]) -> None:
    """Apply a finished sustain's effects on the main thread."""
    if not r.get("slices") and "potion_t" not in r:
        return
    counters["melee"] = counters.get("melee", 0) + int(r.get("swings", 0))
    counters["ranged"] = counters.get("ranged", 0) + int(r.get("volleys", 0))
    counters["sustain_slices"] = counters.get("sustain_slices", 0) + int(r.get("slices", 0))
    counters["sustain_s"] = round(counters.get("sustain_s", 0.0) + float(r.get("seconds", 0)), 2)
    counters["sustain_longest_s"] = max(float(counters.get("sustain_longest_s", 0.0)),
                                        float(r.get("seconds", 0)))
    if "danger_hp" in r:
        counters["sustain_danger_stops"] = counters.get("sustain_danger_stops", 0) + 1
        counters["min_health"] = min(counters.get("min_health", 100), int(r["danger_hp"]))
    if r.get("dodged"):
        counters["dodges"] = counters.get("dodges", 0) + 1
    if "potion_t" in r:
        counters["_last_potion"] = r["potion_t"]
        counters["_potion_pending"] = True
        counters["_potion_hp"] = int(r.get("danger_hp", 0))
        counters["potions"] = counters.get("potions", 0) + 1


SPEED_SUMMARY_KEYS = ("sustain_slices", "sustain_s", "sustain_longest_s",
                      "sustain_danger_stops", "fast_decisions", "full_decisions")


def with_speed(profile: GameProfile, cfg: dict[str, Any]) -> GameProfile:
    """Return the profile with the enabled speed hooks attached."""
    base_summary = profile.summarize

    def summarize(counters: dict[str, Any]) -> dict[str, Any]:
        out = base_summary(counters)
        out.update({k: counters.get(k, 0) for k in SPEED_SUMMARY_KEYS})
        out["speed_settings"] = {k: cfg[k] for k in DEFAULTS}
        return out

    changes: dict[str, Any] = {"summarize": summarize}
    if cfg["sustain_combat"] and cfg["sustain_max_s"] > 0:
        changes.update(sustain=sustain, sustain_max_s=float(cfg["sustain_max_s"]),
                       merge_sustain=merge_sustain)
    if cfg["fast_combat_schema"]:
        changes.update(fast_frame_model=DungeonsCombatDecision,
                       fast_system_prompt=COMBAT_PROMPT,
                       use_fast_decision=make_use_fast(int(cfg["full_refresh_every"])),
                       to_full_decision=to_full_decision)
    if int(cfg["combat_vision_width"]) > 0:
        changes["vision_size"] = make_vision_size(int(cfg["combat_vision_width"]),
                                                  int(cfg["combat_jpeg_quality"]))
    return replace(profile, **changes)
