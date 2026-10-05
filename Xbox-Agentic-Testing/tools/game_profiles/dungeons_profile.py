"""dungeons_profile.py - Minecraft Dungeons half of vision gameplay.

Not yet run on hardware. Bindings come from config/dungeons_controls.yaml
(user transcription of the in-game Controls screen). The shared loop, LLM
plumbing and frame capture live in gameplay_engine.py.

Dungeons is a top-down action game: no F3 coordinates, so progress is judged
by frame change (engine default) and by the model's reading of the scene.
Basic combat reflexes are deterministic (forced_moves) so the model only
steers toward the room-level goal.
"""

from __future__ import annotations

import math
import time
from typing import Any, Literal

import cv2
import numpy as np
from pydantic import BaseModel, Field, PrivateAttr, model_validator

from registry import ToolContext

from combo_dispatch import ComboComponent, dispatch_combo
from dungeons_controls import all_bindings, button_for
from gameplay_engine import GameProfile

_DIAG = 0.7071
_VECTORS: dict[str, tuple[float, float]] = {
    "up": (0.0, -1.0), "down": (0.0, 1.0),
    "left": (-1.0, 0.0), "right": (1.0, 0.0),
    "up_left": (-_DIAG, -_DIAG), "up_right": (_DIAG, -_DIAG),
    "down_left": (-_DIAG, _DIAG), "down_right": (_DIAG, _DIAG),
}
_COMBO_ACTIONS = {"move", "melee", "ranged", "dodge", "forward_dodge", "interact"}
_TAP_ACTIONS = {"artifact_1", "artifact_2", "artifact_3"}
# Short holds for tap-style actions; ranged keeps the model's duration (draw).
_TAP_HOLD = {"melee": 0.25, "dodge": 0.15, "forward_dodge": 0.2, "interact": 0.2}
# Stick held toward the target before/with a tap so the hero faces it.
_FACE_HOLD = 0.15
# Fight burst: several swings in one cycle so enemies don't get free hits
# during the ~5s LLM round-trip.
_FIGHT_SWINGS = 4
_SWING_HOLD = 0.18
_ARTIFACT_EVERY = 3          # fight/ranged cycles between artifact uses
_CROWD_SIZE = 3              # enemies within reach that trigger a retreat
_MAX_RETREAT_RUN = 2         # consecutive retreats before fighting anyway
_RANGED_HOLD = 0.7
# Health monitor: drink (LT) when health is at/below the threshold or fell by
# _POTION_DROP since the last reading. In-game potion cooldown is ~30s.
# Live run 2026-10-05: 61% -> 4% and 68% -> 35% -> 1% within one ~5s cycle each.
_POTION_THRESHOLD = 70       # percent
_POTION_DROP = 20            # percent lost since previous cycle
_POTION_COOLDOWN_S = 30.0    # matches the in-game ~30s potion cooldown
_POTION_HOLD = 0.3
# A drink is only believed once the heart gauge actually refills by this much.
# Until then the cooldown stamp is provisional, so a potion that never landed
# (e.g. the LT press was swallowed) can be retried instead of locking the
# monitor out for a full cooldown.
_POTION_CONFIRM_GAIN = 5     # percent of health regained = drink confirmed
_POTION_RETRY_S = 3.0        # provisional lockout while waiting for the refill
_OPPOSITE = {"up": "down", "down": "up", "left": "right", "right": "left",
             "up_left": "down_right", "down_right": "up_left",
             "up_right": "down_left", "down_left": "up_right"}
_PERPENDICULAR = {"up": "left", "down": "right", "left": "down", "right": "up",
                  "up_left": "down_left", "down_right": "up_right",
                  "up_right": "up_left", "down_left": "down_right"}
_HEADINGS = ("right", "up_right", "up", "up_left",
             "left", "down_left", "down", "down_right")  # 45-degree steps CCW


class _LooseBools(BaseModel):
    """Coerce non-bool model output ('unknown', 'maybe') to False instead of
    failing the whole cycle (a bool_parsing error cost a live cycle)."""

    @model_validator(mode="before")
    @classmethod
    def _coerce_bools(cls, data: Any) -> Any:
        if isinstance(data, dict):
            for name, f in cls.model_fields.items():
                v = data.get(name)
                if v is None:
                    continue
                if f.annotation is bool and not isinstance(v, bool):
                    data[name] = str(v).strip().lower() in ("true", "yes", "y", "1")
                elif f.annotation is int and not isinstance(v, int):
                    # e.g. "3+" or "unknown" cost a live cycle (int_parsing).
                    digits = "".join(ch for ch in str(v) if ch.isdigit())
                    if digits:
                        data[name] = int(digits)
                    else:
                        data.pop(name)
        return data


class DungeonsMove(BaseModel):
    """One controller action. Several may be returned per cycle."""

    action: Literal[
        "move", "melee", "ranged", "dodge", "forward_dodge", "interact",
        "artifact_1", "artifact_2", "artifact_3", "advance_prompt", "wait",
        "fight", "potion",
    ] = Field(description="potion = LT, drink healing potion; fight = burst of melee swings toward `direction` "
                          "then a side dodge; move = left stick; melee = X tap (hero faces "
                          "`direction` first); ranged = hold RT facing "
                          "`direction`; dodge = right-stick flick in "
                          "`direction`; forward_dodge = LB; interact = A "
                          "(jump / open gate, chest, lever); artifact_1/2/3 = "
                          "Y/B/RB; advance_prompt = dismiss a prompt/cutscene; "
                          "wait = do nothing.")
    direction: Literal[
        "up", "down", "left", "right",
        "up_left", "up_right", "down_left", "down_right", "none",
    ] = Field(default="none",
              description="Screen-relative heading ('up' = top of screen). "
                          "For melee/ranged it is the facing toward the target.")
    duration: float = Field(default=0.8, ge=0.1, le=6.0,
                            description="Seconds to hold for move/ranged. "
                                        "Other actions are short taps.")
    purpose: str = Field(default="", description="One line: intended effect.")


class DungeonsEntity(_LooseBools):
    """One thing of interest in the frame. Modelled on minecraft_profile's
    NearbyEntity, but with Dungeons categories and screen-relative 8-way
    headings instead of crosshair-relative ones (the camera is top-down, so
    there is no 'ahead').

    This exists because the objective/loot free-text fields cannot be branched
    on in code: a structured list lets rescue and interaction behaviours be
    arbitrated against combat deterministically.
    """

    category: Literal[
        "enemy", "villager", "loot", "door_or_lever", "objective_marker",
        "building", "portal",
    ] = Field(description="enemy = any hostile; villager = rescuable NPC or "
                          "caged/trapped villager; loot = chest/pickup/emerald; "
                          "door_or_lever = gate, lever, switch, or other "
                          "activatable; objective_marker = the quest marker "
                          "itself; building = village house, hut, farm, "
                          "market stall or other village structure; portal = "
                          "teleport pad / 'Return to' / waypoint.")
    kind: str = Field(description="What it specifically looks like, e.g. "
                                  "'zombie', 'caged villager', 'chest', "
                                  "'iron gate'. Plain words; exact game "
                                  "terminology is not required.")
    direction: Literal[
        "up", "down", "left", "right",
        "up_left", "up_right", "down_left", "down_right", "none",
    ] = Field(default="none",
              description="Screen-relative heading from the hero to it "
                          "('up' = top of screen).")
    distance_estimate: Literal["very_close", "close", "far", "unknown"] = Field(
        default="unknown",
        description="'very_close' = within melee/interaction range now; "
                    "'close' = a second or two of walking; 'far' = distant, "
                    "only useful as a heading.")
    is_threat: bool = Field(
        default=False,
        description="True only for an enemy close enough to matter this "
                    "cycle. False for villagers, loot and distant enemies.")
    threatens_villager: bool = Field(
        default=False,
        description="True for an enemy that is attacking or closing on a "
                    "villager. The system prioritises killing these to "
                    "complete rescue objectives.")


_HEADING = Literal["up", "down", "left", "right",
                   "up_left", "up_right", "down_left", "down_right", "none"]


class AgentCommand(_LooseBools):
    """The Observer's message to one action agent."""

    to: Literal["health", "combat", "interaction", "navigator"] = Field(
        default="navigator",
        description="Which agent should act this cycle.")
    target: str = Field(default="", description="What it is about, e.g. "
                                                "'chest', 'zombie', 'villager', 'objective'.")
    direction: _HEADING = Field(default="none",
                                description="Heading from the hero to the target.")
    reason: str = Field(default="", description="One short line.")


class DungeonsFrameDecision(_LooseBools):
    # Pixel navigation readings set by _annotate (not part of the LLM schema).
    _nav: dict[str, Any] = PrivateAttr(default_factory=dict)
    _agent: str = PrivateAttr(default="")
    _route: str = PrivateAttr(default="")
    scene_state: Literal[
        "in_mission", "menu_or_prompt", "loading_or_cutscene",
        "death", "mission_complete",
    ] = Field(description="Classification of the current screen.")
    player_visible: bool = Field(description="Is the player hero identifiable?")
    enemies: str = Field(description="Visible enemies and where they are "
                                     "relative to the hero; 'none' if none.")
    enemy_adjacent: bool = Field(default=False,
                                 description="An enemy is touching or within "
                                             "~2 hero-widths of the hero.")
    enemy_count: int = Field(default=0, ge=0,
                             description="Number of enemies within ~3 "
                                         "hero-widths of the hero.")
    enemy_direction: Literal[
        "up", "down", "left", "right",
        "up_left", "up_right", "down_left", "down_right", "none",
    ] = Field(default="none", description="Heading from hero to nearest enemy.")
    health_low: bool = Field(default=False,
                             description="Health bar looks nearly empty.")
    health_percent: int = Field(default=100, ge=0, le=100,
                                description="Hero health 0-100, read from the red "
                                            "heart/health gauge at the bottom-centre "
                                            "of the HUD (how full it is).")
    potion_ready: bool = Field(default=True,
                               description="Potion icon next to the health heart "
                                           "is lit (not greyed / on cooldown).")
    objective: str = Field(default="", description="Objective text/marker visible, "
                                                   "and which way it points.")
    loot_or_door: str = Field(default="", description="Chest, door, lever, "
                                                      "or pickup visible and where.")
    entities: list[DungeonsEntity] = Field(
        default_factory=list,
        description="Every notable thing on screen: enemies, villagers, "
                    "chests/pickups, gates/levers, the objective marker. "
                    "This drives rescue and interaction behaviour, so list "
                    "villagers and loot even while enemies are present.")
    villager_in_danger: bool = Field(
        default=False,
        description="A villager is caged, cornered, or under attack and needs "
                    "rescuing now.")
    interact_prompt_visible: bool = Field(
        default=False,
        description="An on-screen button prompt to interact is showing (e.g. "
                    "'Press A to open'), or the hero is standing on a "
                    "highlighted interactable. Means interact would work NOW.")
    objective_kind: Literal[
        "defeat", "find", "rescue", "escort", "activate", "unknown",
    ] = Field(default="unknown",
              description="What the current objective text actually asks for. "
                          "Read it from the objective banner/text, not from "
                          "what enemies happen to be nearby.")
    objective_bearing: Literal[
        "up", "down", "left", "right",
        "up_left", "up_right", "down_left", "down_right", "none",
    ] = Field(default="none",
              description="Direction the objective marker on the minimap "
                          "(top-right) points, as a heading to walk.")
    progress_direction: Literal[
        "up", "down", "left", "right",
        "up_left", "up_right", "down_left", "down_right", "none",
    ] = Field(default="none", description="Heading to make progress.")
    health_mismatch: str = Field(
        default="",
        description="Set by the system, not the model: records when the "
                    "model's health_percent disagreed with the measured "
                    "heart-gauge pixels.")
    reasoning: str = Field(description="ONE short sentence.")
    completion_evidence: str = Field(default="", description="If mission_complete, "
                                     "quote the on-screen evidence.")
    command: AgentCommand = Field(default_factory=AgentCommand,
                                  description="Your message to ONE action agent.")
    # Legacy: the Observer no longer sends moves; kept so old logs/tests load.
    moves: list[DungeonsMove] = Field(default_factory=list,
                                      description="Leave empty.")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


SYSTEM_PROMPT = """\
You are the OBSERVER agent for Minecraft Dungeons running on a real Xbox.
The image is the LIVE screen; report only what is visible. The camera is
top-down isometric. You do NOT press buttons: you describe the screen and
send ONE command to the action agent that should handle it.

ACTION AGENTS (fill `command.to`):
- interaction: an interactable thing is close or a button prompt is showing
  - villager, chest, pickup, gate, lever. It walks there and presses A.
- combat: any enemy is visible. It attacks, shoots and dodges.
- navigator: nothing to fight or use. It follows the minimap to the objective.
- health: only if the hero is clearly about to die.
Villagers and chests are optional: only command interaction when one is
actually on screen. Fill `command.target` and `command.direction` (heading
from the hero), and a short `command.reason`.

OBSERVATION RULES
- Classify the screen first. "YOU ARE DOWNED / REVIVING" = death.
  Mission-complete / results screen = mission_complete, quoting the text.
- Fill enemies / enemy_adjacent / enemy_count / enemy_direction carefully
  (enemy_count = enemies within ~3 hero-widths). Set enemy_direction
  whenever ANY enemy is visible.
- Health is read from the heart gauge by the system; leave health_percent
  at your best guess. Fill potion_ready from the potion icon.
- ALWAYS fill `entities` with everything notable: enemies, villagers,
  chests/pickups, gates/levers, village buildings, portals (teleport pads /
  'Return to' waypoints), the objective marker. Give each a
  distance_estimate (very_close = touching). Set `threatens_villager` on an
  enemy attacking a villager; `villager_in_danger` when a villager is caged,
  cornered or attacked; `interact_prompt_visible` when a button prompt shows.
- Read `objective` and `objective_kind` from the objective banner. Report the
  SAME objective text every cycle until the on-screen text changes.
- The minimap (top-right): white arrow = hero, yellow diamond = objective.
  A flat white/grey silhouette of the hero means a wall is hiding the hero.
- Leave `moves` empty. Keep every text field short.
"""


def _vec(direction: str) -> tuple[float, float]:
    return _VECTORS.get(direction, (0.0, 0.0))


def _bound(action_key: str) -> ComboComponent:
    entry = all_bindings()[action_key]
    return ComboComponent(kind=entry.get("kind", "button"), name=str(entry["button"]))


def _component(move: DungeonsMove) -> tuple[list[ComboComponent], float]:
    comps: list[ComboComponent] = []
    action = move.action
    hold = _TAP_HOLD.get(action, float(move.duration))
    has_dir = move.direction != "none"
    if action == "dodge":
        # The roll is the LB button (bindings: forward_dodge), NOT a right-stick
        # flick - clicking the right stick opens the EMOTE WHEEL (verified on
        # hardware 2026-10-01: rs produced the radial emote menu). The roll
        # travels along the MOVEMENT direction, so hold the left stick toward
        # the target heading while tapping LB.
        if has_dir:
            x, y = _vec(move.direction)
            comps.append(ComboComponent(kind="axis", name="left_stick", x=x, y=y))
        comps.append(_bound("forward_dodge"))
        return comps, hold
    if has_dir and action in ("move", "melee", "ranged"):
        x, y = _vec(move.direction)
        comps.append(ComboComponent(kind="axis", name="left_stick", x=x, y=y))
    if action in ("melee", "ranged", "forward_dodge", "interact"):
        comps.append(_bound(action))
    return comps, hold


def _face(pad: Any, move: DungeonsMove) -> None:
    """Brief left-stick nudge so the hero turns before a melee tap."""
    if move.action == "melee" and move.direction != "none":
        x, y = _vec(move.direction)
        dispatch_combo(pad, [ComboComponent(kind="axis", name="left_stick", x=x, y=y)],
                       hold=_FACE_HOLD, label="face")


def _fight(pad: Any, direction: str) -> dict[str, Any]:
    """Swing repeatedly toward the enemy (stick held so the hero keeps facing
    it), then sidestep with a roll (left stick + LB)."""
    x, y = _vec(direction)
    stick = ComboComponent(kind="axis", name="left_stick", x=x * 0.4, y=y * 0.4)
    swing = [stick, _bound("melee")]
    ok = True
    for _ in range(_FIGHT_SWINGS):
        ok = bool(dispatch_combo(pad, swing, hold=_SWING_HOLD, label="fight")["dispatched"]) and ok
    # Sidestep via the real dodge: left stick toward the sidestep heading plus
    # LB. A right-stick flick here only opened the emote wheel.
    sx, sy = _vec(_PERPENDICULAR.get(direction, "down"))
    dispatch_combo(pad,
                   [ComboComponent(kind="axis", name="left_stick", x=sx, y=sy),
                    _bound("forward_dodge")],
                   hold=0.15, label="fight_dodge")
    return {"macro": "fight", "direction": direction, "swings": _FIGHT_SWINGS,
            "dispatched": ok}


def execute_move(ctx: ToolContext, move: DungeonsMove) -> dict[str, Any]:
    pad = ctx.hardware.pad()
    action = move.action
    if action == "fight":
        if move.direction == "none":
            return {"macro": action, "dispatched": False, "error": "no direction given"}
        return _fight(pad, move.direction)
    if action in _COMBO_ACTIONS:
        comps, hold = _component(move)
        if not comps:
            return {"macro": action, "dispatched": False,
                    "error": "no direction given"}
        _face(pad, move)
        res = dispatch_combo(pad, comps, hold=hold, label=action)
        return {"macro": action, "direction": move.direction, "duration": hold,
                "dispatched": bool(res["dispatched"])}
    if action == "potion":
        res = dispatch_combo(pad, [_bound("potion")], hold=_POTION_HOLD, label="potion")
        return {"macro": action, "button": button_for("potion"),
                "dispatched": bool(res["dispatched"])}
    if action in _TAP_ACTIONS:
        button = button_for(action)
        ok = pad.press(button, duration=0.2)
        return {"macro": action, "button": button, "dispatched": bool(ok)}
    if action == "advance_prompt":
        button = button_for("interact")
        ok = pad.press(button, duration=0.2)
        time.sleep(0.3)
        return {"macro": action, "button": button, "dispatched": bool(ok)}
    if action == "wait":
        time.sleep(min(float(move.duration), 3.0))
        return {"macro": "wait", "dispatched": False}
    return {"macro": action, "dispatched": False,
            "error": f"'{action}' is not implemented."}


def execute_moves(ctx: ToolContext, moves: list[DungeonsMove]) -> list[dict[str, Any]]:
    """Merge combo moves into one simultaneous GIMX call when no control is
    used twice (e.g. move + forward_dodge); otherwise run them in order."""
    if len(moves) < 2 or not all(m.action in _COMBO_ACTIONS for m in moves):
        return [execute_move(ctx, m) for m in moves]
    comps: list[ComboComponent] = []
    hold = 0.0
    for m in moves:
        c, h = _component(m)
        comps += c
        hold = max(hold, h)
    if len({c.name for c in comps}) != len(comps):
        return [execute_move(ctx, m) for m in moves]
    res = dispatch_combo(ctx.hardware.pad(), comps, hold=min(hold, 6.0),
                         label="+".join(m.action for m in moves))
    return [{"macro": m.action, "direction": m.direction,
             "dispatched": bool(res["dispatched"])} for m in moves]


def _initial_counters() -> dict[str, Any]:
    return {"deaths": 0, "melee": 0, "dodges": 0, "reflex_attacks": 0,
            "stuck_streak": 0, "ranged": 0, "artifacts": 0,
            "retreats": 0, "_retreat_run": 0,
            "combat_cycles": 0, "artifact_idx": 0, "_downed": False,
            "potions": 0, "_last_potion": float("-inf"), "_last_health": 100,
            "min_health": 100, "_potion_pending": False, "_potion_hp": 100,
            "potions_confirmed": 0, "rescue_attacks": 0, "rescue_interacts": 0,
            "rescue_approaches": 0, "model_yields": 0, "interactions": 0,
            "health_mismatches": 0, "wall_escapes": 0, "map_steers": 0,
            "occluded_cycles": 0, "map_travel_px": 0, "_blocked_streak": 0,
            "_escape_idx": 0, "_nav_grace": 0, "_detour": None}


def _next_artifact(counters: dict[str, Any]) -> DungeonsMove:
    idx = counters["artifact_idx"] % 3
    counters["artifact_idx"] += 1
    return DungeonsMove(action=f"artifact_{idx + 1}", purpose="Use artifact in combat.")


# Heart gauge (bottom-centre HUD) box as fractions of a 1920x1080 frame;
# calibrated from live frames: a full heart fills ~93 of the 140 rows red,
# draining as health drops, 0 when downed.
_HEART_Y = (900 / 1080, 1040 / 1080)
_HEART_X = (620 / 1920, 820 / 1920)
_HEART_FULL_ROWS = 93


def _pixel_health(frame: Any) -> int | None:
    """Health % from the red fill of the heart gauge (BGR frame)."""
    if frame is None or getattr(frame, "ndim", 0) != 3:
        return None
    h, w = frame.shape[:2]
    box = frame[int(h * _HEART_Y[0]):int(h * _HEART_Y[1]),
                int(w * _HEART_X[0]):int(w * _HEART_X[1])].astype("float32")
    b, g, r = box[..., 0], box[..., 1], box[..., 2]
    red = (r > 120) & (r > g * 1.6) & (r > b * 1.6)
    rows = int((red.mean(axis=1) > 0.2).sum())
    full = _HEART_FULL_ROWS * box.shape[0] / 140
    return round(min(100.0, rows / full * 100))


# Model-vs-pixel health gap (percentage points) worth reporting. Frame 017
# had the model claim "critically low (30%)" while the gauge read 100, and the
# overwrite hid it - so disagreements are now logged instead of swallowed.
_HEALTH_DISAGREE = 20


# Minimap box and hero-centre box as fractions of the frame, calibrated on live
# 1920x1080 frames (2026-10-05).
_MAP_BOX = (1630 / 1920, 15 / 1080, 1890 / 1920, 275 / 1080)
_HERO_BOX = (800 / 1920, 380 / 1080, 1120 / 1920, 700 / 1080)
# Hero hidden by a wall renders as a flat grey silhouette: 700-3700 px blobs
# while stuck, <=200 px in normal play.
_OCCLUDED_MIN_PX = 500
# Minimap scroll per cycle with overlays masked: 0-3 px when stuck, 8-75 px
# while walking (live run 2026-10-05).
_MAP_STILL_PX = 4.0
_MAP_MOVED_PX = 8.0
_MAP_WINDOW = None
_BLOCKED_CYCLES = 2          # still-map cycles before escaping (1 if occluded)
_MARKER_REACHED_PX = 12      # marker this close to the arrow = arrived
_ESCAPE_OFFSETS = (90, -90, 135, -135, 180)
_prev_map: list[Any] = [None]


def _box(frame: Any, box: tuple[float, float, float, float]) -> Any:
    h, w = frame.shape[:2]
    return frame[int(h * box[1]):int(h * box[3]), int(w * box[0]):int(w * box[2])]


def _heading_of(dx: float, dy: float) -> str:
    ang = math.degrees(math.atan2(-dy, dx)) % 360
    return _HEADINGS[int(round(ang / 45)) % 8]


def _rotate(heading: str, degrees: int) -> str:
    return _HEADINGS[(_HEADINGS.index(heading) + degrees // 45) % 8]


def _angle_diff(a: str, b: str) -> int:
    d = abs(_HEADINGS.index(a) - _HEADINGS.index(b)) % 8
    return min(d, 8 - d) * 45


def _map_bearing(crop: Any) -> tuple[str | None, float | None]:
    """Heading from the white hero arrow to the yellow objective diamond."""
    c = crop.astype(np.int32)
    b, g, r = c[..., 0], c[..., 1], c[..., 2]
    ys, xs = np.nonzero((r > 190) & (g > 160) & (b < 90))
    if len(xs) < 30:
        return None, None
    h, w = crop.shape[:2]
    px, py = w / 2, h / 2
    mid = c[h // 3:2 * h // 3, w // 3:2 * w // 3]
    wy, wx = np.nonzero((mid > 225).all(axis=2))
    if len(wx):
        px, py = wx.mean() + w // 3, wy.mean() + h // 3
    dx, dy = xs.mean() - px, ys.mean() - py
    dist = math.hypot(dx, dy)
    if dist < _MARKER_REACHED_PX:
        return None, dist
    return _heading_of(dx, dy), dist


def _hero_occluded(frame: Any) -> bool:
    c = _box(frame, _HERO_BOX).astype(np.int32)
    flat = (((c.max(axis=2) - c.min(axis=2)) < 16) & (c.min(axis=2) > 140)).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(flat)
    return n > 1 and int(stats[1:, cv2.CC_STAT_AREA].max()) >= _OCCLUDED_MIN_PX


def _map_texture(crop: Any) -> Any:
    """Scrolling map terrain only: the border, objective diamond (top edge),
    corner icon and the always-centred hero arrow are cut or flattened, else
    they pin the correlation at zero shift."""
    global _MAP_WINDOW
    h, w = crop.shape[:2]
    inner = crop[int(h * 0.17):int(h * 0.83), int(w * 0.06):int(w * 0.94)]
    g = cv2.cvtColor(inner, cv2.COLOR_BGR2GRAY).astype(np.float32)
    ih, iw = g.shape
    cy, cx = int(h * 0.5) - int(h * 0.17), int(w * 0.5) - int(w * 0.06)
    g[max(0, cy - 15):cy + 15, max(0, cx - 15):cx + 15] = g.mean()
    g = cv2.GaussianBlur(g, (3, 3), 0)
    if _MAP_WINDOW is None or _MAP_WINDOW.shape != g.shape:
        _MAP_WINDOW = cv2.createHanningWindow((iw, ih), cv2.CV_32F)
    return g - g.mean()


def _read_nav(frame: Any) -> dict[str, Any]:
    crop = _box(frame, _MAP_BOX)
    tex = _map_texture(crop)
    shift = None
    prev = _prev_map[0]
    if prev is not None and prev.shape == tex.shape:
        (sx, sy), _ = cv2.phaseCorrelate(prev, tex, _MAP_WINDOW)
        shift = round(math.hypot(sx, sy), 1)
    _prev_map[0] = tex
    bearing, dist = _map_bearing(crop)
    return {"bearing": bearing, "marker_px": None if dist is None else round(dist),
            "shift": shift, "occluded": _hero_occluded(frame)}


def _annotate(decision: DungeonsFrameDecision, frame: Any) -> None:
    if frame is not None and getattr(frame, "ndim", 0) == 3:
        nav = _read_nav(frame)
        decision._nav = nav
        if nav["bearing"]:
            decision.objective_bearing = nav["bearing"]
    hp = _pixel_health(frame)
    if hp is None:
        return
    claimed = decision.health_percent
    if abs(claimed - hp) >= _HEALTH_DISAGREE:
        decision.health_mismatch = (
            f"model said {claimed}%, heart gauge reads {hp}% (using {hp}%)")
        print(f"  [health] model {claimed}% vs pixel {hp}% -> trusting pixel",
              flush=True)
    # The pixel reading stays authoritative: it is measured, not inferred.
    decision.health_percent = hp
    if hp > 0:
        decision.health_low = hp <= _POTION_THRESHOLD
    decision._nav["hud"] = _hud_visible(frame)


# Blue artifact hexagon right of the hotbar: 0.39-0.73 blue fill in gameplay,
# 0.0 on loading screens. The heart can't be used - it empties at low health.
_HUD_HEX_BOX = (1140 / 1920, 920 / 1080, 1210 / 1920, 1010 / 1080)
_HUD_MIN_BLUE = 0.2


def _hud_visible(frame: Any) -> bool:
    if frame is None or getattr(frame, "ndim", 0) != 3:
        return False
    c = _box(frame, _HUD_HEX_BOX).astype(np.int32)
    b, g, r = c[..., 0], c[..., 1], c[..., 2]
    return float(((b > 180) & (b > r + 60) & (g > 100)).mean()) >= _HUD_MIN_BLUE


def _confirm_success(decision: DungeonsFrameDecision, counters: dict[str, Any]) -> bool:
    """Reject 'mission complete' while the gameplay HUD is still on screen: a
    new objective banner ('Return to the Farmer') was misread as the end."""
    if decision._nav.get("hud"):
        counters["rejected_completions"] = counters.get("rejected_completions", 0) + 1
        print("  [verify] mission_complete rejected: gameplay HUD still visible",
              flush=True)
        return False
    return True


def _potion_check(decision: DungeonsFrameDecision,
                  counters: dict[str, Any]) -> list[DungeonsMove] | None:
    """Health monitor: drink when health is low or just dropped sharply."""
    hp = decision.health_percent
    if decision.health_low:
        hp = min(hp, _POTION_THRESHOLD - 1)
    prev = counters["_last_health"]
    # Update the baseline BEFORE the hp == 0 bail-out. Returning early used to
    # leave _last_health stale at the pre-death value, so the first real
    # reading after a revive looked like a huge drop and burned a potion.
    if hp > 0:
        counters["_last_health"] = hp
        counters["min_health"] = min(counters["min_health"], hp)
    # Confirm a previous drink actually landed: health climbing back up is the
    # only real evidence. Once confirmed, the stamp becomes the full cooldown.
    if counters.get("_potion_pending") and hp - counters.get("_potion_hp", 0) >= _POTION_CONFIRM_GAIN:
        counters["_potion_pending"] = False
        counters["potions_confirmed"] = counters.get("potions_confirmed", 0) + 1
    if hp == 0:
        return None  # downed or HUD hidden: a potion would be wasted
    low = hp <= _POTION_THRESHOLD
    dropped = prev - hp >= _POTION_DROP and hp < 100
    # An unconfirmed drink only blocks for a short retry window, not 30s.
    wait_s = _POTION_RETRY_S if counters.get("_potion_pending") else _POTION_COOLDOWN_S
    cooled = time.monotonic() - counters["_last_potion"] >= wait_s
    if not (low or dropped) or not cooled or not decision.potion_ready:
        return None
    counters["_last_potion"] = time.monotonic()
    counters["_potion_pending"] = True
    counters["_potion_hp"] = hp
    why = f"health {hp}% (was {prev}%)"
    moves = [DungeonsMove(action="potion", purpose=f"Drink potion: {why}.")]
    enemy_dir = decision.enemy_direction
    if decision.enemy_adjacent and enemy_dir != "none":
        away = _OPPOSITE.get(enemy_dir, "down")
        moves += [DungeonsMove(action="dodge", direction=away,
                               purpose="Potion: dodge away while healing."),
                  DungeonsMove(action="move", direction=away, duration=1.0,
                               purpose="Potion: open distance.")]
    elif enemy_dir == "none":
        moves += [m for m in decision.moves if m.action == "move"][:1]
    return moves


def _entities(decision: DungeonsFrameDecision,
              category: str) -> list[DungeonsEntity]:
    return [e for e in decision.entities if e.category == category]


def _villager_threat_dir(decision: DungeonsFrameDecision) -> str | None:
    """Heading to the enemy attacking a villager, if any.

    Rescue objectives fail when the bot keeps hitting whatever is nearest to
    the hero instead of whatever is nearest to the villager, so this target
    selection deliberately ignores `enemy_direction`.
    """
    for e in _entities(decision, "enemy"):
        if e.threatens_villager and e.direction != "none":
            return e.direction
    return None


def _interaction_target(decision: DungeonsFrameDecision) -> DungeonsEntity | None:
    """A loot/door entity that is close enough to act on this cycle."""
    for cat in ("door_or_lever", "loot"):
        for e in _entities(decision, cat):
            if e.distance_estimate in ("very_close", "close"):
                return e
    return None


def _yield_to_model(decision: DungeonsFrameDecision) -> bool:
    """True when the model's own plan should run even though enemies are on
    screen.

    The combat reflex used to hijack EVERY cycle in which any enemy was
    visible (the only escape was `enemy_direction == "none"`), so interaction,
    rescue approach and objective navigation never got a turn. This mirrors
    the hatch minecraft_profile._village_forced_moves already uses. Genuine
    danger - an adjacent enemy or low health - still keeps the reflex.
    """
    if decision.enemy_adjacent or decision.health_low:
        return False
    if decision.interact_prompt_visible:
        return True
    # Standing next to the thing the objective is about: let the model finish.
    if decision.objective_kind in ("activate", "find", "rescue"):
        target = _interaction_target(decision)
        if target is not None and target.distance_estimate == "very_close":
            return True
    return False


def _rescue_moves(decision: DungeonsFrameDecision,
                  counters: dict[str, Any]) -> list[DungeonsMove] | None:
    """Kill the enemy threatening a villager, or close on the villager."""
    if not decision.villager_in_danger:
        return None
    threat_dir = _villager_threat_dir(decision)
    if threat_dir is not None:
        counters["rescue_attacks"] = counters.get("rescue_attacks", 0) + 1
        return [DungeonsMove(action="fight", direction=threat_dir,
                             purpose="Rescue: kill the enemy on the villager.")]
    villagers = [e for e in _entities(decision, "villager") if e.direction != "none"]
    if not villagers:
        return None
    v = villagers[0]
    if v.distance_estimate == "very_close":
        counters["rescue_interacts"] = counters.get("rescue_interacts", 0) + 1
        return [DungeonsMove(action="interact", direction=v.direction,
                             purpose="Rescue: free the villager.")]
    counters["rescue_approaches"] = counters.get("rescue_approaches", 0) + 1
    return [DungeonsMove(action="move", direction=v.direction, duration=1.2,
                         purpose="Rescue: close on the villager.")]


def _update_nav(decision: DungeonsFrameDecision, counters: dict[str, Any]) -> None:
    """Track whether the last walk actually moved the hero on the minimap."""
    nav = decision._nav
    shift = nav.get("shift")
    counters["_prev_move_dir"] = counters.pop("_last_move_dir", None)
    if nav.get("occluded"):
        counters["occluded_cycles"] += 1
    if shift is not None:
        counters["map_travel_px"] += int(shift)
        if shift >= _MAP_MOVED_PX:
            counters["_escape_idx"] = 0
    # The minimap read lags the input by about a cycle (pipelined frames), so
    # an escape gets a grace period before being judged.
    if counters["_nav_grace"] > 0:
        counters["_nav_grace"] -= 1
        counters["_blocked_streak"] = 0
        return
    if counters["_prev_move_dir"] and shift is not None and shift < _MAP_STILL_PX:
        counters["_blocked_streak"] += 1
    else:
        counters["_blocked_streak"] = 0


def _nav_moves(decision: DungeonsFrameDecision,
               counters: dict[str, Any]) -> list[DungeonsMove] | None:
    """Wall escape, then minimap steering. None = keep the model's moves."""
    nav = decision._nav
    last = counters.get("_prev_move_dir")
    need = 1 if nav.get("occluded") else _BLOCKED_CYCLES
    if last and counters["_blocked_streak"] >= need:
        off = _ESCAPE_OFFSETS[counters["_escape_idx"] % len(_ESCAPE_OFFSETS)]
        counters["_escape_idx"] += 1
        counters["_blocked_streak"] = 0
        counters["_nav_grace"] = 2
        counters["wall_escapes"] += 1
        heading = _rotate(last, off)
        counters["_detour"] = heading
        why = "hero hidden behind wall" if nav.get("occluded") else "minimap not moving"
        return [DungeonsMove(action="dodge", direction=heading,
                             purpose=f"Blocked going {last} ({why}): roll {heading}."),
                DungeonsMove(action="move", direction=heading, duration=2.0,
                             purpose="Blocked: walk around the wall.")]
    detour = counters.get("_detour")
    if detour:
        counters["_detour"] = None
        bearing = nav.get("bearing")
        # Keep sliding sideways, but bend toward the objective when possible.
        heading = detour
        if bearing and _angle_diff(detour, bearing) == 90:
            heading = _rotate(detour, 45 if _rotate(detour, 90) == bearing else -45)
        return [DungeonsMove(action="move", direction=heading, duration=2.0,
                             purpose="Detour: clear the wall before heading back.")]
    bearing = nav.get("bearing")
    if not bearing or not decision.moves:
        return None
    out, changed = [], False
    for m in decision.moves:
        if m.action == "move" and m.direction != "none" and _angle_diff(m.direction, bearing) > 90:
            out.append(m.model_copy(update={
                "direction": bearing,
                "purpose": f"Minimap: objective is {bearing}, not {m.direction}."}))
            changed = True
        else:
            out.append(m)
    if not changed:
        return None
    counters["map_steers"] += 1
    return out


def _forced_moves(decision: DungeonsFrameDecision,
                  counters: dict[str, Any]) -> list[DungeonsMove] | None:
    state = decision.scene_state
    if state == "death":
        # Count each downing once, not every cycle of the revive countdown.
        if not counters.get("_downed"):
            counters["deaths"] += 1
        counters["_downed"] = True
        return [DungeonsMove(action="wait", duration=3.0,
                             purpose="Downed: auto-revive countdown.")]
    counters["_downed"] = False
    if state == "loading_or_cutscene":
        return [DungeonsMove(action="wait", duration=1.5, purpose="Wait out loading.")]
    if state == "menu_or_prompt" and not decision.moves:
        return [DungeonsMove(action="advance_prompt", purpose="Dismiss prompt.")]
    if state != "in_mission":
        return None
    _update_nav(decision, counters)
    enemy_dir = decision.enemy_direction
    # Priority order: survival (potion) > rescue > yield-to-model > combat.
    potion = _potion_check(decision, counters)
    if potion:
        return potion
    rescue = _rescue_moves(decision, counters)
    if rescue:
        return rescue
    if _yield_to_model(decision) and decision.moves:
        counters["model_yields"] = counters.get("model_yields", 0) + 1
        return None
    if enemy_dir == "none":
        return _nav_moves(decision, counters)
    return _combat_moves(decision, counters)


def _combat_moves(decision: DungeonsFrameDecision,
                  counters: dict[str, Any]) -> list[DungeonsMove]:
    """Reflex combat toward `enemy_direction` (must not be 'none')."""
    enemy_dir = decision.enemy_direction
    counters["combat_cycles"] += 1
    use_artifact = counters["combat_cycles"] % _ARTIFACT_EVERY == 1
    if decision.health_low and decision.enemy_adjacent:
        return [_next_artifact(counters),
                DungeonsMove(action="dodge", direction=_OPPOSITE.get(enemy_dir, "down"),
                             purpose="Low health: dodge away."),
                DungeonsMove(action="move", direction=_OPPOSITE.get(enemy_dir, "down"),
                             duration=1.0, purpose="Low health: retreat.")]
    # Crowd: 3+ enemies close -> break away and shoot back so they string out
    # into a line, then fight. Capped at _MAX_RETREAT_RUN consecutive retreats
    # so a cornered hero still fights instead of kiting into a wall forever.
    if (decision.enemy_adjacent and decision.enemy_count >= _CROWD_SIZE
            and counters["_retreat_run"] < _MAX_RETREAT_RUN):
        counters["_retreat_run"] += 1
        counters["retreats"] += 1
        away = _OPPOSITE.get(enemy_dir, "down")
        return [DungeonsMove(action="dodge", direction=away,
                             purpose=f"Crowd of {decision.enemy_count}: dodge out."),
                DungeonsMove(action="move", direction=away, duration=1.0,
                             purpose="Crowd: open distance."),
                DungeonsMove(action="ranged", direction=enemy_dir, duration=_RANGED_HOLD,
                             purpose="Crowd: shoot the chasers.")]
    counters["_retreat_run"] = 0
    if decision.enemy_adjacent:
        counters["reflex_attacks"] += 1
        moves = [DungeonsMove(action="fight", direction=enemy_dir,
                              purpose="Enemy in reach: swing burst + sidestep.")]
        if use_artifact:
            moves.insert(0, _next_artifact(counters))
        return moves
    # Enemy visible but not adjacent: shoot, then let the model's moves follow
    # on alternate cycles so the hero still advances.
    moves = [DungeonsMove(action="ranged", direction=enemy_dir, duration=_RANGED_HOLD,
                          purpose="Enemy at range: shoot.")]
    if use_artifact:
        moves.insert(0, _next_artifact(counters))
    if counters["combat_cycles"] % 2 == 0:
        moves += [m for m in decision.moves if m.action == "move"][:1]
    return moves[:3]


def _fallback_moves(decision: DungeonsFrameDecision) -> list[DungeonsMove]:
    # Prefer the minimap objective bearing over progress_direction: the runs
    # showed the objective flip-flopping (Honeycomb Farm <-> Brave Haven)
    # because navigation followed per-cycle prose instead of the marker.
    for heading in (decision.objective_bearing, decision.progress_direction):
        if heading != "none":
            return [DungeonsMove(action="move", direction=heading, duration=0.8,
                                 purpose="No move returned; advance.")]
    return [DungeonsMove(action="move", direction="up", duration=0.8,
                         purpose="No move returned; advance.")]


def _on_move_dispatched(move: DungeonsMove, counters: dict[str, Any]) -> None:
    if move.action == "move" and move.direction != "none":
        counters["_last_move_dir"] = move.direction
    if move.action == "melee":
        counters["melee"] += 1
    elif move.action == "fight":
        counters["melee"] += _FIGHT_SWINGS
        counters["dodges"] += 1
    elif move.action in ("dodge", "forward_dodge"):
        counters["dodges"] += 1
    elif move.action == "ranged":
        counters["ranged"] += 1
    elif move.action in _TAP_ACTIONS:
        counters["artifacts"] += 1
    elif move.action == "potion":
        counters["potions"] += 1
    elif move.action == "interact":
        counters["interactions"] = counters.get("interactions", 0) + 1


def _print_extra(decision: DungeonsFrameDecision) -> None:
    print(f"  Enemies   : {decision.enemies} (adjacent={decision.enemy_adjacent}, "
          f"near={decision.enemy_count})", flush=True)
    print(f"  Health    : {decision.health_percent}% (potion_ready={decision.potion_ready})",
          flush=True)
    print(f"  Objective : {decision.objective} "
          f"(kind={decision.objective_kind}, bearing={decision.objective_bearing})",
          flush=True)
    print(f"  Loot/door : {decision.loot_or_door}", flush=True)
    if decision.entities:
        summary = ", ".join(f"{e.category}:{e.kind}@{e.direction}"
                            for e in decision.entities[:6])
        print(f"  Entities  : {summary}", flush=True)
    if decision.villager_in_danger:
        print("  Rescue    : villager in danger", flush=True)
    if decision.interact_prompt_visible:
        print("  Interact  : prompt visible", flush=True)
    nav = decision._nav
    if nav:
        print(f"  Minimap   : objective={nav.get('bearing')} "
              f"(marker {nav.get('marker_px')}px) shift={nav.get('shift')}px "
              f"wall_glow={nav.get('occluded')}", flush=True)


def _cycle_extra_fields(decision: DungeonsFrameDecision) -> dict[str, Any]:
    return {"enemy_count": decision.enemy_count, "enemies": decision.enemies, "enemy_adjacent": decision.enemy_adjacent,
            "health_low": decision.health_low, "health_percent": decision.health_percent,
            "potion_ready": decision.potion_ready, "objective": decision.objective,
            "loot_or_door": decision.loot_or_door,
            "objective_kind": decision.objective_kind,
            "objective_bearing": decision.objective_bearing,
            "villager_in_danger": decision.villager_in_danger,
            "interact_prompt_visible": decision.interact_prompt_visible,
            "health_mismatch": decision.health_mismatch,
            "map_bearing": decision._nav.get("bearing"),
            "map_shift": decision._nav.get("shift"),
            "hero_occluded": decision._nav.get("occluded"),
            "entities": [e.model_dump() for e in decision.entities]}


def _terminal_evidence(decision: DungeonsFrameDecision, cycle: int,
                       frame_path: str) -> dict[str, Any]:
    return {"cycle": cycle, "scene_state": decision.scene_state,
            "evidence": decision.completion_evidence, "frame_path": frame_path}


_SUMMARY_KEYS = ("deaths", "melee", "dodges", "reflex_attacks", "ranged",
                 "artifacts", "retreats", "potions", "min_health",
                 "potions_confirmed", "rescue_attacks", "rescue_interacts",
                 "rescue_approaches", "model_yields", "interactions",
                 "wall_escapes", "map_steers", "occluded_cycles", "map_travel_px")


def _extra_session_text(counters: dict[str, Any]) -> str:
    return ("Deaths so far: {deaths}; melee swings: {melee}; ranged: {ranged}; "
            "artifacts: {artifacts}; dodges: {dodges}; potions: {potions}\n").format(
                **{k: counters.get(k, 0) for k in _SUMMARY_KEYS})


def _summarize(counters: dict[str, Any]) -> dict[str, Any]:
    return {k: counters.get(k, 0) for k in _SUMMARY_KEYS}


PROFILE = GameProfile(
    key="dungeons",
    move_model=DungeonsMove,
    frame_model=DungeonsFrameDecision,
    system_prompt=SYSTEM_PROMPT,
    default_goal=("Follow the objective marker through the mission, fight "
                  "enemies that block the way, open chests, and reach the "
                  "mission-complete screen."),
    artifact_prefix="dungeons",
    json_artifact_name="dungeons-cycles.json",
    execute_move=execute_move,
    execute_moves=execute_moves,
    success_states={"mission_complete"},
    terminal_flag_key="mission_complete",
    terminal_evidence_key="mission_evidence",
    build_terminal_evidence=_terminal_evidence,
    initial_counters=_initial_counters,
    forced_moves=_forced_moves,
    fallback_moves=_fallback_moves,
    on_move_dispatched=_on_move_dispatched,
    extra_session_text=_extra_session_text,
    print_extra=_print_extra,
    annotate_decision=_annotate,
    confirm_success=_confirm_success,
    cycle_extra_fields=_cycle_extra_fields,
    summarize=_summarize,
)
