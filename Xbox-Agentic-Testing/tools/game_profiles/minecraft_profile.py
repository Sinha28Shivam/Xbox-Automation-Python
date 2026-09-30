"""minecraft_profile.py - the Minecraft-specific half of vision gameplay.

Everything here is genuinely game-specific: the move vocabulary, the frame
schema, the system prompt describing Bedrock's crafting UI, move dispatch to
the real pad, and the mine_streak/stuck_streak split (mining's screen deltas
are hardware-measured to be much smaller than movement/look deltas, so it
gets its own patience counter - see the note on `_update_stuck_counters`).
The shared loop, LLM plumbing and frame capture live in gameplay_engine.py.
"""

from __future__ import annotations

import time
from typing import Any, Literal

from pydantic import BaseModel, Field

from registry import ToolContext

from combo_dispatch import ComboComponent, dispatch_combo, resolve_component
from gameplay_engine import GameProfile
from minecraft_controls import button_for


# ===========================================================================
# What the model is allowed to decide
# ===========================================================================
class MinecraftMove(BaseModel):
    """One controller action. Several may be returned per cycle."""

    action: Literal[
        "move", "run", "look", "mine", "attack", "jump", "sprint", "interact",
        "press_button", "navigate_recipe", "craft_all", "wait",
        "use", "quick_move", "loot_chest",
    ] = Field(description="The controller action to execute. 'use' taps LT "
                          "(Use Item) to open a chest/villager; 'quick_move' "
                          "presses Y to transfer the highlighted stack/trade "
                          "on a chest or trade screen.")

    direction: Literal["left", "right", "up", "down", "none"] = Field(
        default="none",
        description="Travel direction for move, camera-turn direction for "
                    "look, or D-pad direction for navigate_recipe. UP = "
                    "forward/walk-ahead, DOWN = backward - there is no "
                    "separate 'forward'/'backward' value, only up/down (a "
                    "real run rejected direction='forward' with a "
                    "validation error and lost that cycle). The 'All "
                    "recipes' list is a HORIZONTAL ROW of icons, so use "
                    "'left'/'right' here - NOT up/down, which do nothing on "
                    "this screen (hardware-verified: up/down produced zero "
                    "screen change).")

    duration: float = Field(
        default=0.8, ge=0.1, le=8.0,
        description="Seconds to hold the stick/trigger/button. For 'run' "
                    "use 3-6 (sustained sprint). For 'mine' "
                    "this is how long RT stays held on the block - a single "
                    "log usually needs several seconds of continuous holding. "
                    "For 'attack' this is IGNORED - each attack is always a "
                    "short tap (repeated swings need repeated attack moves, "
                    "not one long hold), UNVERIFIED on hardware against a "
                    "real mob as of this field's introduction.")

    button: str = Field(
        default="",
        description="interact/press_button override, e.g. 'a', 'x', 'rb', 'lb'. "
                    "Use 'x' to open the Crafting screen (game's own label - "
                    "NOT a separate 'inventory' screen, confirmed via the "
                    "live game's Settings -> Controller -> Button Mapping), "
                    "'b' to close it, 'rb'/'lb' to cycle hotbar slots. Not "
                    "used by navigate_recipe or craft_all, which always use "
                    "D-pad and Y respectively.")

    purpose: str = Field(
        default="",
        description="One line: what this move is meant to achieve on screen.")


class NearbyEntity(BaseModel):
    """One thing of interest visible in the frame - mob, structure, or
    resource. Generic on purpose: combat needs mobs, village-looting needs
    structures, gathering needs resources, and a single list avoids three
    near-identical schemas that would drift apart the way the old
    per-game gameplay loops did before the gameplay_engine refactor.
    """

    category: Literal["mob", "structure", "resource"] = Field(
        description="mob = any living creature (hostile or passive); "
                    "structure = village/building/chest/crafting station; "
                    "resource = tree/ore/water/other gatherable material.")

    kind: str = Field(
        description="What it specifically looks like, e.g. 'zombie', 'cow', "
                    "'wooden house', 'chest', 'oak tree', 'iron ore vein'. "
                    "Best guess in plain words - exact game terminology is "
                    "not required.")

    direction: Literal["left", "right", "ahead", "behind_hint", "unknown"] = Field(
        default="unknown",
        description="Roughly where it is relative to the crosshair/screen "
                    "centre. 'behind_hint' is for something only partially "
                    "visible at a screen edge, suggesting it continues "
                    "outside the frame behind the player.")

    distance_estimate: Literal["very_close", "close", "far", "unknown"] = Field(
        default="unknown",
        description="'very_close' = within melee/interaction range now; "
                    "'close' = a few seconds of walking; 'far' = distant, "
                    "only useful as a heading to remember.")

    is_threat: bool = Field(
        default=False,
        description="True only for a mob that is hostile AND close enough to "
                    "matter this cycle (e.g. a nearby zombie/skeleton). False "
                    "for passive mobs, distant mobs, structures, and "
                    "resources.")


class MinecraftFrameDecision(BaseModel):
    """The model's structured reading of one gameplay frame, plus its plan."""

    scene_state: Literal[
        "in_gameplay", "menu_or_prompt", "cutscene_or_loading",
        "inventory_open", "planks_crafted", "stuck_or_blocked",
        "chest_open", "trade_open",
    ] = Field(description="Classification of the current screen.")

    player_visible: bool = Field(
        description="Is the first-person view / hotbar / crosshair visible?")

    # Village-loot fields (default False so the planks profile is unaffected).
    chest_visible: bool = Field(
        default=False,
        description="A chest/barrel block is visible in the 3D view.")
    villager_visible: bool = Field(
        default=False,
        description="A villager (NPC with big nose) is visible in the 3D view.")
    loot_taken: bool = Field(
        default=False,
        description="Only on a chest screen: your PREVIOUS quick_move visibly "
                    "removed a stack from the chest grid.")
    trade_done: bool = Field(
        default=False,
        description="Only on a trade screen: your PREVIOUS action visibly "
                    "completed a trade (payment consumed / result received).")

    tree_visible: bool = Field(
        default=False,
        description="Is a tree trunk visible anywhere in the frame?")
    log_visible: bool = Field(
        default=False,
        description="Is a wood log BLOCK (not a mob) visible directly in "
                    "front of the crosshair, close enough to mine? A real "
                    "hardware run mined a creeper standing at the crosshair "
                    "for 2.5s after misreading its dark, vertical silhouette "
                    "as a log trunk in dim/night lighting - if ANYTHING that "
                    "moves, has legs, or could be a mob is at the crosshair, "
                    "this must be False even if it looks vaguely trunk-like.")
    mob_blocking_crosshair: bool = Field(
        default=False,
        description="Is a mob (any kind, hostile or passive) standing at or "
                    "very near the crosshair, in the spot you might "
                    "otherwise mine? Check this BEFORE setting log_visible - "
                    "log_visible and mob_blocking_crosshair should never "
                    "both be true for the same thing at the crosshair.")

    nearby_entities: list[NearbyEntity] = Field(
        default_factory=list,
        description="Every mob, structure, or resource worth noting in this "
                    "frame - not just trees/logs, which have their own "
                    "dedicated fields above for the current tree-chopping "
                    "goal. Empty list if nothing else of interest is visible. "
                    "This does not change what moves are valid this cycle - "
                    "it is observational, for awareness and later recall.")

    recipe_highlighted: str = Field(
        default="",
        description="When scene_state is inventory_open: the name/description "
                    "of whichever recipe is currently highlighted/selected in "
                    "the 'All recipes' row (e.g. 'Planks', 'Stick', 'Crafting "
                    "Table'). Empty if the inventory is not open or nothing is "
                    "clearly highlighted yet.")

    craftable_count: int = Field(
        default=0,
        description="The craftable quantity badge shown directly on a recipe "
                    "icon in the 'All recipes' row (a small number in the "
                    "corner, e.g. the Planks icon showing '4') - this is "
                    "visible on the icon itself, no highlight/selection is "
                    "needed to read it. 0 if no badge is visible, meaning "
                    "that recipe cannot be made yet from current inventory.")

    reasoning: str = Field(
        description="ONE short sentence: goal + why these moves achieve it. "
                    "A fast reaction beats a thorough essay - a real player "
                    "does not narrate before every button press, so do not "
                    "either. No step-by-step breakdown.")

    moves: list[MinecraftMove] = Field(
        default_factory=list,
        description="1-3 moves to run this cycle, in order. Prefer ONE move "
                    "when the situation is uncertain so the next frame shows "
                    "its isolated effect.")

    confidence: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Confidence that these moves make progress toward "
                    "crafting planks.")


# ===========================================================================
# System prompt
# ===========================================================================
MINECRAFT_GAMEPLAY_SYSTEM_PROMPT = """\
You are playing Minecraft (Xbox Bedrock Edition) on a real Xbox One, in
survival first-person view, inside a freshly created world. You control it
through an emulated controller. The attached image is the LIVE screen right
now - reason only from what you can actually see in it.

GOAL: find the nearest tree, break its logs, open the inventory, and craft
the logs into planks. Then stop - do not build, explore further, or fight
anything once planks exist in the inventory/crafting grid.

EXPLORATION STRATEGY: do NOT creep in 0.8s steps. If no tree is visible,
use `run` (duration 4-6) in a straight line, and `look` to pick a new
heading. Trees are green canopies on brown trunks - ignore houses, fences
and villagers. If a wall, house or fence fills the view, `look` 1-2s to
turn well away from it, then `run` again. Once a tree is in view, `run`
toward it, then switch to short `move` steps and `mine` when the trunk is
under the crosshair.

If the view is very dark or enclosed (dark brick, cave, no sky), you are
inside a structure or hole: turn around and sprint back the way you came,
or jump up out of it - do not keep mining the walls. Only `mine` LOG blocks
(brown bark / tree trunks), never bricks, stone or dirt. The loop will
automatically open the crafting screen after several mining cycles; when it
is open, read the Planks badge into `craftable_count` and, if > 0, use
craft_all.

While pursuing that goal, also report any mob, structure, or resource you
notice in `nearby_entities` (e.g. a zombie, a distant house, an ore vein) -
this is purely observational and does not change which moves are valid this
cycle. Only set `is_threat` true for a hostile mob close enough to matter
right now; do not flag distant or passive mobs as threats.

CONTROLS AVAILABLE TO YOU (as macros)
  move          : left stick, short walk step in `direction` (forward=up,
                  back=down). Use only for small corrections near a target.
  run           : SPRINT forward for a long stretch (duration 3-6s): holds
                  the left stick up and clicks LS so the player sprints
                  continuously. THIS is how you explore - one run covers
                  30-60 blocks. Can be combined with look (turn while
                  running) and jump (hop obstacles) in the same cycle.
  look          : right stick, turn the camera in `direction` to find/face a tree
  mine          : hold RT while facing a block, breaks it after a few seconds
                  of continuous holding. Must be standing close enough that
                  the block is directly under the crosshair. NEVER mine if
                  a mob is at the crosshair (a real run mistook a creeper's
                  dark vertical silhouette for a log trunk at night and held
                  RT on it) - check `mob_blocking_crosshair` first.
  jump          : press A once - clear a 1-block step or gap.
  sprint        : click the left stick (LS) to toggle sprinting while moving.
                  Combine with move in the SAME cycle for a running jump or
                  a fast retreat - do not send sprint alone with no move.
  navigate_recipe: press D-pad LEFT/RIGHT to move the highlight between
                  recipes in the 'All recipes' row on the left of the
                  inventory screen. This row is HORIZONTAL, not vertical -
                  up/down do nothing here (hardware-verified: zero screen
                  change when tried).
  craft_all     : press Y to craft the MAXIMUM amount of the currently
                  highlighted recipe in one action. This is the ONLY way to
                  actually produce items - selecting a recipe alone does not
                  craft it.
  interact      : a single 'a' press - use ONLY to open/close a sub-menu or
                  confirm a non-crafting prompt. Pressing 'a' on a recipe
                  does NOT craft it on this UI - use craft_all (Y) instead.
  press_button  : a single button press with an explicit `button`, e.g. 'x'
                  to open the Crafting screen (the game's own name for it -
                  confirmed via Settings -> Controller -> Button Mapping,
                  NOT a separate 'inventory' screen), 'b' to close it
                  (NOT 'x' again - 'x' only opens, it does not toggle
                  closed), 'rb'/'lb' to cycle hotbar slots
  wait          : do nothing this cycle, for a screen that is still loading

THE CRAFTING WORKFLOW ON THIS UI (Xbox Bedrock 'All recipes' panel):
  The recipe list is a HORIZONTAL ROW of icons near the top-left of the
  inventory screen, each with a small number badge in its corner showing
  how many you can craft right now from your inventory (0 if you are
  missing an ingredient). You do NOT need to highlight an icon to read its
  badge - read the badges directly off the icons in the frame.
  1. Find the recipe icon you want (e.g. Planks) in the row and read its
     badge number directly - report it as `craftable_count` and the
     recipe's name as `recipe_highlighted`.
  2. If its badge is 0, you are missing an ingredient (usually: no raw log
     in inventory yet - go mine one first).
  3. If you need to move the SELECTION cursor onto that icon (e.g. because
     a later step needs it focused), use navigate_recipe with direction
     'left' or 'right' - NEVER up/down, which do nothing on this row.
  4. Once the recipe you want shows a badge > 0, use craft_all (Y) to craft
     the maximum amount in one press. Do NOT press 'a'/interact repeatedly
     on a recipe - that does not craft anything on this screen and was
     measured stalling an entire run for 6+ cycles.
  NOTE: there is also a row of TABS above the recipe row (LB/RB cycle between
  tabs like the magnifying-glass 'All recipes' search tab, armor, etc.) - do
  not confuse that tab bar with the recipe row itself.

SCENE STATE RULES
  in_gameplay      : normal first-person view, hotbar/crosshair visible, no
                      menu overlay
  menu_or_prompt    : any overlay, dialog, or prompt is on screen
  cutscene_or_loading: a loading/black/transition screen with no player HUD
  inventory_open    : the inventory/crafting screen is open
  planks_crafted     : planks are visibly present in the inventory or
                        hotbar - THIS IS THE SUCCESS STATE, report it as soon
                        as you can see plank items on screen
  stuck_or_blocked   : repeated attempts are not changing the screen at all

Return your reading of the frame and 1-3 moves for this cycle.
"""


# ===========================================================================
# Move execution - all button/stick names resolved via the shared pad
# ===========================================================================
def execute_minecraft_move(ctx: ToolContext, move: MinecraftMove) -> dict[str, Any]:
    """Dispatch one decided move. Returns what was sent, never a verdict."""
    pad = ctx.hardware.pad()
    action = move.action
    duration = float(move.duration)

    if action == "move":
        heading = move.direction if move.direction != "none" else "up"
        dispatched = pad.stick("left_stick", direction=heading,
                               duration=duration, strength=1.0)
        return {"macro": "move", "direction": heading,
                "duration": duration, "dispatched": bool(dispatched)}

    if action == "run":
        return _dispatch_run(pad, [move])[0]

    if action == "look":
        heading = move.direction if move.direction != "none" else "right"
        dispatched = pad.stick("right_stick", direction=heading,
                               duration=duration, strength=0.6)
        return {"macro": "look", "direction": heading,
                "duration": duration, "dispatched": bool(dispatched)}

    if action == "mine":
        # button_for("attack_destroy") -> "rt", read directly from the
        # game's own Settings -> Controller -> Button Mapping screen
        # (config/minecraft_controls.yaml), not assumed. Also matches the
        # hardware-verified trigger semantics used elsewhere: RT is a
        # 0..32767 axis on XOnePad, not 0..255, so pad.hold("rt", ...)
        # (which resolves to the trigger's configured default_press) is
        # what actually registers a full pull on the console.
        rt = button_for("attack_destroy")
        dispatched = pad.hold(rt, max(0.3, min(8.0, duration)))
        return {"macro": "mine", "button": rt,
                "duration": duration, "dispatched": bool(dispatched)}

    if action == "attack":
        # button_for("attack_destroy") -> "rt", CONFIRMED by the game's own
        # button-mapping screen (2026-09-25) - Attack/Destroy really is RT,
        # so a real hardware run's failure to land a hit is NOT because
        # this action targets the wrong physical button; the bug (see
        # repo memory: first combat run, player died, 0 attacks
        # dispatched) was in the combat LOOP never getting close enough to
        # use this action at all. A short, fixed tap (0.25s) regardless of
        # the requested duration, since repeated swings need repeated
        # attack moves, not one long hold (a long RT hold instead starts
        # mining/destroying whatever is under the crosshair - same
        # physical trigger, tap vs hold is what the game's own mapping
        # screen groups under one "Attack / Destroy" entry).
        rt = button_for("attack_destroy")
        dispatched = pad.hold(rt, 0.25)
        return {"macro": "attack", "button": rt,
                "duration": 0.25, "dispatched": bool(dispatched)}

    if action == "interact":
        button = move.button or "a"
        dispatched = pad.press(button, duration=min(duration, 0.30))
        return {"macro": "interact", "button": button,
                "dispatched": bool(dispatched)}

    if action == "press_button":
        # Default "x" CONFIRMED by the game's own button-mapping screen:
        # X = "Crafting" in world-gameplay context - this is the crafting/
        # "All recipes" screen this loop actually navigates, not a
        # separate generic "inventory" screen (Y opens a DIFFERENT screen
        # per the same mapping - an earlier version of this comment/route
        # step imprecisely called X's screen "inventory"; the buttons
        # pressed were always correct, only the label was loose).
        button = move.button or button_for("crafting")
        dispatched = pad.press(button, duration=min(duration, 0.30))
        return {"macro": "press_button", "button": button,
                "dispatched": bool(dispatched)}

    if action == "navigate_recipe":
        # The 'All recipes' row is horizontal (hardware-verified: up/down
        # produced zero screen change), so 'right' is the sane default -
        # not 'down', which does nothing on this screen.
        heading = move.direction if move.direction != "none" else "right"
        dispatched = pad.press(heading, duration=min(duration, 0.20))
        return {"macro": "navigate_recipe", "direction": heading,
                "dispatched": bool(dispatched)}

    if action == "craft_all":
        # Bedrock's 'All recipes' panel crafts the maximum makeable amount
        # of the currently highlighted recipe on a single Y press - there is
        # no separate "confirm quantity" step.
        dispatched = pad.press("y", duration=min(duration, 0.30))
        return {"macro": "craft_all", "button": "y",
                "dispatched": bool(dispatched)}

    if action == "use":
        # LT = Use Item / Place Block (minecraft_controls.yaml). A short
        # pull opens chests and villager trade screens.
        lt = button_for("use_item_place_block")
        dispatched = pad.hold(lt, max(0.2, min(1.0, duration)))
        return {"macro": "use", "button": lt, "dispatched": bool(dispatched)}

    if action == "loot_chest":
        from .chest_loot import loot_chest_grid
        res = loot_chest_grid(ctx)
        print(f"[chest-loot] {res}", flush=True)
        c = _ACTIVE_COUNTERS
        if c is not None:
            c["loot_verified_transfers"] = c.get("loot_verified_transfers", 0) + res["verified_transfers"]
            c["loot_failed_transfers"] = c.get("loot_failed_transfers", 0) + res["failed_transfers"]
            c["loot_runs"] = c.get("loot_runs", 0) + 1
            if res["initial_occupied"] and not res["remaining_occupied"]:
                c["chest_emptied"] = True
        return res

    if action == "quick_move":
        dispatched = pad.press("y", duration=min(duration, 0.30))
        return {"macro": "quick_move", "button": "y",
                "dispatched": bool(dispatched)}

    if action == "jump":
        dispatched = pad.press("a", duration=min(duration, 0.25))
        return {"macro": "jump", "button": "a", "dispatched": bool(dispatched)}

    if action == "sprint":
        dispatched = pad.press("ls", duration=min(duration, 0.15))
        return {"macro": "sprint", "button": "ls", "dispatched": bool(dispatched)}

    if action == "wait":
        time.sleep(min(duration, 3.0))
        return {"macro": "wait", "duration": duration, "dispatched": False}

    return {"macro": action, "dispatched": False,
            "error": f"'{action}' is not an implemented Minecraft move."}


# ===========================================================================
# Simultaneous dispatch - move+look+attack(+jump/sprint) together, like a
# real player's hands work at once, instead of one sequential subprocess
# call per action. The generic send/hold/release plumbing lives in
# combo_dispatch.py (shared with every other game); this function only maps
# Minecraft's own move vocabulary onto that engine's ComboComponent list.
# ===========================================================================
_SIMULTANEOUS_COMPATIBLE = {"move", "run", "look", "attack", "mine", "jump", "sprint"}

_RUN_MAX_SECONDS = 8.0
_SPRINT_CLICK_SECONDS = 0.2  # LS is a toggle: click once at run start, then release

_LOOK_STRENGTH = 0.6  # gentler than a full deflection, or the camera whips too fast to track a target


def _minecraft_move_to_component(move: MinecraftMove) -> tuple[ComboComponent, float]:
    """One MinecraftMove -> (ComboComponent, hold duration for that component)."""
    if move.action == "run":
        heading = move.direction if move.direction in ("up", "down", "left", "right") else "up"
        hold = max(0.5, min(_RUN_MAX_SECONDS, float(move.duration)))
        return ComboComponent(kind="stick", name="left_stick", direction=heading), hold
    if move.action == "move":
        heading = move.direction if move.direction != "none" else "up"
        return ComboComponent(kind="stick", name="left_stick", direction=heading), float(move.duration)
    if move.action == "look":
        heading = move.direction if move.direction != "none" else "right"
        return ComboComponent(kind="stick", name="right_stick", direction=heading,
                              strength=_LOOK_STRENGTH), float(move.duration)
    if move.action in ("attack", "mine"):
        hold = 0.25 if move.action == "attack" else max(0.3, min(8.0, move.duration))
        return ComboComponent(kind="trigger", name="rt"), hold
    if move.action == "jump":
        return ComboComponent(kind="button", name="a"), min(float(move.duration), 0.25)
    if move.action == "sprint":
        return ComboComponent(kind="button", name="ls"), min(float(move.duration), 0.15)
    raise ValueError(f"'{move.action}' is not simultaneous-dispatch compatible")


def _dispatch_run(pad: Any, moves: list[MinecraftMove]) -> list[dict[str, Any]]:
    """Sustained sprint: every component is pressed together, each is
    released at its own hold time (LS click and jump are short, the run
    stick and look are long)."""
    sender = getattr(pad, "_send_events", None)
    if not callable(sender):
        return [{"macro": m.action, "dispatched": False,
                 "error": "pad has no _send_events"} for m in moves]

    timed: list[tuple[float, list[tuple[str, int]]]] = []
    press_all: list[tuple[str, int]] = []
    outcomes: list[dict[str, Any]] = []
    sprint_wanted = False
    for move in moves:
        component, hold = _minecraft_move_to_component(move)
        press, release = resolve_component(pad, component)
        press_all += press
        timed.append((hold, release))
        outcome: dict[str, Any] = {"macro": move.action, "duration": hold}
        if component.kind == "stick":
            outcome["direction"] = component.direction
        else:
            outcome["button"] = component.name
        outcomes.append(outcome)
        if move.action == "run" and component.direction == "up":
            sprint_wanted = True
    if sprint_wanted:
        press, release = resolve_component(pad, ComboComponent(kind="button", name="ls"))
        press_all += press
        timed.append((_SPRINT_CLICK_SECONDS, release))

    ok = bool(sender(press_all, "run:press"))
    elapsed = 0.0
    for hold, release in sorted(timed, key=lambda t: t[0]):
        time.sleep(max(0.0, hold - elapsed))
        elapsed = max(elapsed, hold)
        ok = bool(sender(release, "run:release")) and ok
    for o in outcomes:
        o["dispatched"] = ok
    return outcomes


def execute_minecraft_moves(ctx: ToolContext,
                            moves: list[MinecraftMove]) -> list[dict[str, Any]]:
    """Dispatch up to 3 moves this cycle - SIMULTANEOUSLY when they are all
    move/look/attack/mine/jump/sprint (the combinations a real player's
    hands actually use at once: sprint+jump, move+look, mine+look, ...),
    falling back to the old one-at-a-time execute_minecraft_move for
    anything else (menu navigation, crafting - actions that make no sense
    to combine).

    Returns one outcome dict per move, in the same order and same shape as
    execute_minecraft_move's return value, so callers do not need to know
    which path was taken.
    """
    if not moves:
        return []

    if not all(m.action in _SIMULTANEOUS_COMPATIBLE for m in moves) or len(moves) < 2:
        return [execute_minecraft_move(ctx, m) for m in moves]

    pad = ctx.hardware.pad()
    if any(m.action == "run" for m in moves):
        first = moves[0]
        if first.action == "look" and first.duration >= 1.0 and len(moves) > 1:
            # Long turn first, so the run starts on the new heading.
            return [execute_minecraft_move(ctx, first)] + _dispatch_run(pad, moves[1:])
        return _dispatch_run(pad, moves)

    components: list[ComboComponent] = []
    outcomes: list[dict[str, Any]] = []
    max_hold = 0.0

    for move in moves:
        component, hold = _minecraft_move_to_component(move)
        components.append(component)
        max_hold = max(max_hold, hold)
        outcome: dict[str, Any] = {"macro": move.action, "dispatched": True}
        if component.kind == "stick":
            outcome["direction"] = component.direction
            outcome["duration"] = move.duration
        else:
            outcome["button"] = component.name
            outcome["duration"] = hold
        outcomes.append(outcome)

    result = dispatch_combo(pad, components, hold=max(0.1, min(8.0, max_hold)),
                            label="+".join(m.action for m in moves))
    for o in outcomes:
        o["dispatched"] = result["dispatched"]

    return outcomes


# ===========================================================================
# Fallbacks / forced moves for particular scene states
# ===========================================================================
def _fallback_moves(decision: MinecraftFrameDecision) -> list[MinecraftMove]:
    if decision.scene_state == "cutscene_or_loading":
        return [MinecraftMove(action="wait", duration=1.0,
                              purpose="Wait out a loading/transition screen.")]
    if decision.scene_state == "menu_or_prompt":
        return [MinecraftMove(action="interact", button="a",
                              purpose="Dismiss the prompt/menu.")]
    if decision.scene_state == "inventory_open":
        # Browsing the recipe list is always safe here, unlike spamming 'a'
        # on a recipe (which does not craft anything on this UI and was
        # measured stalling entire runs for 6+ cycles). 'right' matches the
        # row's horizontal layout - hardware-verified 'down' produces zero
        # screen change here.
        return [MinecraftMove(
            action="navigate_recipe", direction="right", duration=0.2,
            purpose="Model returned no move; browse the recipe list.")]
    if decision.scene_state == "chest_open":
        return [MinecraftMove(action="loot_chest", purpose="Chest open: loot every occupied slot with Y.")]
    if decision.scene_state == "trade_open":
        return [MinecraftMove(action="interact", button="a", purpose="Trade screen: select the highlighted trade.")]
    # No decision is still a decision: turn and sprint to explore rather
    # than burning a cycle standing still.
    return [
        MinecraftMove(action="look", direction="right", duration=0.6,
                      purpose="Model returned no move; turn to a new heading."),
        MinecraftMove(action="run", direction="up", duration=4.0,
                      purpose="Model returned no move; sprint to explore."),
    ]


def _print_extra(decision: MinecraftFrameDecision) -> None:
    print(f"  Tree seen : {decision.tree_visible}", flush=True)
    print(f"  Log seen  : {decision.log_visible}"
         f"{'  !! MOB AT CROSSHAIR' if decision.mob_blocking_crosshair else ''}",
         flush=True)
    if decision.scene_state == "inventory_open":
        print(f"  Recipe    : {decision.recipe_highlighted or '(none)'} "
              f"x{decision.craftable_count}", flush=True)
    if (decision.chest_visible or decision.villager_visible or decision.loot_taken
            or decision.trade_done or decision.scene_state in ("chest_open", "trade_open")):
        print(f"  Village   : chest={decision.chest_visible} villager={decision.villager_visible} "
              f"loot_taken={decision.loot_taken} trade_done={decision.trade_done}", flush=True)
    if decision.nearby_entities:
        summary = ", ".join(
            f"{e.kind}({e.category}/{e.direction}/{e.distance_estimate}"
            f"{'/THREAT' if e.is_threat else ''})"
            for e in decision.nearby_entities)
        print(f"  Nearby    : {summary}", flush=True)


def _cycle_extra_fields(decision: MinecraftFrameDecision) -> dict[str, Any]:
    return {
        "tree_visible": decision.tree_visible,
        "log_visible": decision.log_visible,
        "mob_blocking_crosshair": decision.mob_blocking_crosshair,
        "chest_visible": decision.chest_visible,
        "villager_visible": decision.villager_visible,
        "loot_taken": decision.loot_taken,
        "trade_done": decision.trade_done,
        "recipe_highlighted": decision.recipe_highlighted,
        "craftable_count": decision.craftable_count,
        "nearby_entities": [e.model_dump() for e in decision.nearby_entities],
    }


def _build_terminal_evidence(decision: MinecraftFrameDecision, cycle: int,
                             frame_path: str | None) -> dict[str, Any]:
    return {"cycle": cycle, "reasoning": decision.reasoning,
            "frame_path": frame_path}


# ===========================================================================
# Stuck-counter grouping
# ===========================================================================
# RCA fix (run-20260924-142031): the crack-overlay on a block being mined
# covers a small part of the frame, unlike a camera turn or a walk step.
# Hardware-measured deltas of 0.011-0.4 were recorded while a log was
# genuinely being chipped away (it appeared in the hotbar two cycles later),
# but idle/no-op steps elsewhere in the SAME run showed deltas up to 0.071
# from sensor noise alone, so no single delta threshold reliably tells real
# mining progress apart from noise. Mining gets its OWN patience counter,
# 2x the general limit, and never feeds the general stuck_streak.
def _update_stuck_counters(counters: dict[str, int], moves: list[MinecraftMove],
                           progressed: bool) -> None:
    was_mining = any(m.action == "mine" for m in moves)
    if was_mining:
        counters["mine_cycles"] = counters.get("mine_cycles", 0) + 1
        counters["mine_streak"] = 0 if progressed else counters.get("mine_streak", 0) + 1
    else:
        counters["stuck_streak"] = 0 if progressed else counters.get("stuck_streak", 0) + 1
        counters["mine_streak"] = 0


def _stuck_limits(stuck_limit: int) -> dict[str, int]:
    base = max(2, int(stuck_limit))
    # Escape maneuvers start at streak 3, so the stop must sit well above that.
    return {"stuck_streak": max(10, base * 2), "mine_streak": base * 2}


def _stuck_message(name: str, value: int) -> str:
    if name == "mine_streak":
        return (f"{value} consecutive mine attempts produced no visual "
                f"change even at a relaxed threshold. Either input is not "
                f"reaching the console or the model is mining an "
                f"unreachable/incorrect block. Stopping instead of sending "
                f"more input into a void.")
    return (f"{value} consecutive non-mining cycles produced no visual "
            f"change even with escape maneuvers. The player is most likely "
            f"boxed in by terrain or structures (or the screen is frozen). "
            f"Stopping instead of sending more input into a void.")


# ===========================================================================
# Forced re-aim after repeated failed mine attempts
# ===========================================================================
# REAL BUG FOUND ON HARDWARE (2026-09-28): a 12-cycle live run repeatedly
# reported log_visible=True and dispatched mine, but the crosshair was
# actually resting on leaf canopy/gaps between trunks (confirmed by
# inspecting the saved frames), not a log block - so RT was held on empty
# air over and over. mine_streak already tracks exactly this ("N failed
# mine attempts in a row"), but nothing was using it to actually CHANGE
# the approach - the model kept re-guessing the same wrong aim. Forcing a
# small deliberate look-down + step-forward after 2 failures breaks the
# loop by moving the crosshair to a genuinely different spot, instead of
# hoping the model's next guess is better.
_REAIM_AFTER_STREAK = 2
_ESCAPE_AFTER_STREAK = 3
_CRAFT_CHECK_AFTER_MINES = 5


def _forced_moves(decision: MinecraftFrameDecision,
                  counters: dict[str, Any]) -> list[MinecraftMove] | None:
    # SAFETY OVERRIDE, checked first: never mine a mob. A real hardware run
    # held RT on a creeper for 2.5s after the model itself said log_visible
    # (the schema field didn't exist yet to catch this) - this is not left
    # to the model's own judgment call the way the re-aim below is, because
    # a creeper standing that close is a real explosion risk in survival.
    if decision.mob_blocking_crosshair:
        return [MinecraftMove(
            action="look", direction="right", duration=0.5,
            purpose="SAFETY OVERRIDE: a mob is at the crosshair, not a log - "
                    "turning away instead of mining it.")]

    # Crafting screen handling: bound how long it stays open.
    if decision.scene_state == "inventory_open":
        counters["inv_cycles"] = counters.get("inv_cycles", 0) + 1
        if counters["inv_cycles"] >= 3 and decision.craftable_count <= 0:
            counters["inv_cycles"] = 0
            counters["mine_cycles"] = 0
            return [MinecraftMove(
                action="press_button", button="b", duration=0.2,
                purpose="Crafting check found no craftable planks - close it and "
                        "go chop another tree.")]
        return None
    counters["inv_cycles"] = 0

    # Mining cap: after N mining cycles, check the crafting screen for a Planks badge.
    if (counters.get("mine_cycles", 0) >= _CRAFT_CHECK_AFTER_MINES
            and decision.scene_state == "in_gameplay"):
        counters["mine_cycles"] = 0
        counters["craft_checks"] = counters.get("craft_checks", 0) + 1
        return [MinecraftMove(
            action="press_button", button="x", duration=0.2,
            purpose="MINING CAP: open crafting to check for a Planks badge.")]

    streak = counters.get("stuck_streak", 0)
    if streak >= _ESCAPE_AFTER_STREAK and decision.scene_state in (
            "in_gameplay", "stuck_or_blocked"):
        counters["escapes"] = counters.get("escapes", 0) + 1
        n = counters["escapes"]
        turn = "right" if n % 2 else "left"
        variant = n % 4
        if variant == 0:  # back out, turn hard, sprint
            return [
                MinecraftMove(action="move", direction="down", duration=2.0,
                              purpose="ESCAPE: back away from the obstacle."),
                MinecraftMove(action="look", direction=turn, duration=2.2,
                              purpose="ESCAPE: turn well away."),
                MinecraftMove(action="run", direction="up", duration=5.0,
                              purpose="ESCAPE: sprint along the new heading."),
            ]
        if variant == 2:  # about-face and sprint back the way we came
            return [
                MinecraftMove(action="look", direction=turn, duration=4.0,
                              purpose="ESCAPE: about-face (~180 deg)."),
                MinecraftMove(action="run", direction="up", duration=6.0,
                              purpose="ESCAPE: sprint back the way we came."),
            ]
        if variant == 3:  # sidestep, then hop and sprint
            return [
                MinecraftMove(action="move", direction=turn, duration=1.5,
                              purpose="ESCAPE: sidestep along the wall."),
                MinecraftMove(action="jump", duration=0.25,
                              purpose="ESCAPE: hop a fence/ledge."),
                MinecraftMove(action="run", direction="up", duration=4.0,
                              purpose="ESCAPE: sprint forward."),
            ]
        return [
            MinecraftMove(action="look", direction=turn,
                          duration=2.2 if streak >= 5 else 1.1,
                          purpose="ESCAPE: no progress - turn away from the obstacle."),
            MinecraftMove(action="run", direction="up", duration=4.0,
                          purpose="ESCAPE: sprint along the new heading."),
            MinecraftMove(action="jump", duration=0.25,
                          purpose="ESCAPE: hop a fence/ledge if one is in the way."),
        ]

    if counters.get("mine_streak", 0) >= _REAIM_AFTER_STREAK:
        return [
            MinecraftMove(action="look", direction="down", duration=0.3,
                         purpose="FORCED RE-AIM: repeated mine attempts produced "
                                 "no change - the crosshair is likely on leaves/"
                                 "air, not the trunk. Look down slightly."),
            MinecraftMove(action="move", direction="up", duration=0.4,
                         purpose="FORCED RE-AIM: step closer so the trunk fills "
                                 "more of the frame under the corrected crosshair."),
        ]
    return None


# ===========================================================================
# Village profile: explore, loot a chest, trade with a villager
# ===========================================================================
VILLAGE_SYSTEM_PROMPT = """\
You are playing Minecraft (Xbox Bedrock) on a real Xbox One, survival, first
person, standing in or near a VILLAGE. The attached image is the LIVE screen.
Reason only from what you can see.

GOAL: (1) find a CHEST (brown wooden box with a metal latch, often inside
houses) and loot it; (2) find a VILLAGER (NPC with a big nose, robe) and open
their trade screen and complete one trade. Do NOT ignore chests or
villagers - they are the targets. Do NOT mine anything.

SET FLAGS HONESTLY: chest_visible / villager_visible only when you clearly
see one. loot_taken only on a chest screen when the chest grid visibly lost a
stack after your last quick_move. trade_done only on a trade screen when the
previous action visibly consumed payment or produced a result item.

EXPLORING: use `run` (3-5s) along streets, `look` to choose a heading. If a
wall/fence fills the view, `look` 1-2s away then `run`. Jump (A) hops fences
and 1-block steps. If the view is dark or enclosed, back out and turn around.
Avoid pits, water, lava.

APPROACH: when a chest or villager is visible, face it (`look`), `run` or
`move` up until it is very close (distance_estimate very_close), keep it at
the crosshair, then `use` (taps LT) to open it. Do not `use` from far away.
A villager standing at the crosshair is NOT a mob to avoid.

ON A CHEST SCREEN (scene_state=chest_open): the chest grid is the TOP panel;
your inventory is below. The loop transfers items with quick_move (Y) and
D-pad navigation and closes it with B - you may also request those moves.
ON A TRADE SCREEN (scene_state=trade_open): trades are listed on the left,
selected trade's payment and result slots on the right. Use D-pad via
navigate_recipe (up/down) to pick a trade, interact 'a' to select it,
quick_move (Y) to take the result; 'b' closes it.

CONTROLS: move, run (sprint), look, jump, use (LT tap: open chest/villager),
quick_move (Y), interact (single button, set `button`), navigate_recipe
(D-pad, `direction`), press_button (`button`, e.g. 'b' to close a screen),
wait. Never use `mine` or `attack`.

SCENE STATES: in_gameplay, menu_or_prompt, cutscene_or_loading,
inventory_open (the player's own inventory/crafting screen - close with b),
chest_open, trade_open, stuck_or_blocked (nothing changes).
(planks_crafted is unused here - never report it.)

Return your reading of the frame and 1-3 moves.
"""

_SCREEN_MAX_CYCLES = 5
_AFTER_SCREEN_COOLDOWN = 4
_ACTIVE_COUNTERS: dict[str, Any] | None = None


def _village_forced_moves(decision: MinecraftFrameDecision,
                          counters: dict[str, Any]) -> list[MinecraftMove] | None:
    global _ACTIVE_COUNTERS
    _ACTIVE_COUNTERS = counters
    state = decision.scene_state
    if state == "chest_open":
        counters["chest_cycles"] = counters.get("chest_cycles", 0) + 1
        counters["screen_seen_chest"] = counters.get("screen_seen_chest", 0) + 1
        n = counters["chest_cycles"]
        if n > 2 or counters.get("chest_emptied"):
            counters["chest_emptied"] = False
            counters["chest_cycles"] = 0
            counters["cooldown"] = _AFTER_SCREEN_COOLDOWN
            return [MinecraftMove(action="press_button", button="b", duration=0.2,
                                  purpose="Chest looted/checked long enough - close it.")]
        return [MinecraftMove(action="loot_chest",
                              purpose="Walk the chest grid and Y-transfer every occupied slot.")]
    counters["chest_cycles"] = 0

    if state == "trade_open":
        counters["trade_cycles"] = counters.get("trade_cycles", 0) + 1
        counters["screen_seen_trade"] = counters.get("screen_seen_trade", 0) + 1
        n = counters["trade_cycles"]
        if n > _SCREEN_MAX_CYCLES:
            counters["trade_cycles"] = 0
            counters["cooldown"] = _AFTER_SCREEN_COOLDOWN
            return [MinecraftMove(action="press_button", button="b", duration=0.2,
                                  purpose="Trade attempts done - close the screen.")]
        if n == 1:
            return [MinecraftMove(action="interact", button="a",
                                  purpose="Select the highlighted trade."),
                    MinecraftMove(action="quick_move", purpose="Take the trade result.")]
        heading = "down" if n % 2 == 0 else "up"
        return [MinecraftMove(action="navigate_recipe", direction=heading, duration=0.2,
                              purpose="Try the next trade in the list."),
                MinecraftMove(action="interact", button="a", purpose="Select that trade."),
                MinecraftMove(action="quick_move", purpose="Take the trade result.")]
    counters["trade_cycles"] = 0

    if state == "inventory_open":
        return [MinecraftMove(action="press_button", button="b", duration=0.2,
                              purpose="Wrong screen - close the inventory.")]

    if counters.get("cooldown", 0) > 0 and state == "in_gameplay":
        counters["cooldown"] -= 1
        return [MinecraftMove(action="look", direction="right", duration=1.5,
                              purpose="Leave the just-used chest/villager - turn away."),
                MinecraftMove(action="run", direction="up", duration=4.0,
                              purpose="Explore onward to the next target.")]

    # A villager/chest at the crosshair is a target, not a hazard.
    if decision.mob_blocking_crosshair and (decision.villager_visible or decision.chest_visible):
        return None
    return _forced_moves(decision, counters)


def _village_confirm_success(decision: MinecraftFrameDecision,
                             counters: dict[str, Any]) -> bool:
    """Called only on chest_open/trade_open frames. Counts the model's
    claims, and ends the run only once BOTH a loot and a trade are claimed."""
    if decision.scene_state == "chest_open" and decision.loot_taken:
        counters["loot_claims"] = counters.get("loot_claims", 0) + 1
    if decision.scene_state == "trade_open" and decision.trade_done:
        counters["trade_claims"] = counters.get("trade_claims", 0) + 1
    # Loot success is MEASURED (chest slot emptied after Y), not model-claimed.
    return (counters.get("loot_verified_transfers", 0) >= 1
            and counters.get("trade_claims", 0) >= 1)


def _village_terminal_evidence(decision: MinecraftFrameDecision, cycle: int,
                               frame_path: str | None) -> dict[str, Any]:
    return {"cycle": cycle, "reasoning": decision.reasoning,
            "frame_path": frame_path,
            "note": "MODEL CLAIM of loot+trade; verify the frame pairs."}


def _village_summarize(counters: dict[str, Any]) -> dict[str, Any]:
    return {k: counters.get(k, 0) for k in (
        "screen_seen_chest", "screen_seen_trade", "loot_claims", "trade_claims",
        "loot_verified_transfers", "loot_failed_transfers", "loot_runs")}


VILLAGE_PROFILE = GameProfile(
    key="minecraft-village",
    move_model=MinecraftMove,
    frame_model=MinecraftFrameDecision,
    system_prompt=VILLAGE_SYSTEM_PROMPT,
    default_goal=("Find a chest and loot it, then find a villager and complete "
                  "one trade."),
    artifact_prefix="village",
    json_artifact_name="minecraft-village-cycles.json",
    execute_move=execute_minecraft_move,
    execute_moves=execute_minecraft_moves,
    success_states={"chest_open", "trade_open"},
    terminal_flag_key="loot_and_trade_claimed",
    terminal_evidence_key="loot_and_trade_evidence",
    build_terminal_evidence=_village_terminal_evidence,
    confirm_success=_village_confirm_success,
    fallback_moves=_fallback_moves,
    forced_moves=_village_forced_moves,
    update_stuck_counters=_update_stuck_counters,
    stuck_limits=_stuck_limits,
    stuck_message=_stuck_message,
    print_extra=_print_extra,
    cycle_extra_fields=_cycle_extra_fields,
    summarize=_village_summarize,
)


PROFILE = GameProfile(
    key="minecraft",
    move_model=MinecraftMove,
    frame_model=MinecraftFrameDecision,
    system_prompt=MINECRAFT_GAMEPLAY_SYSTEM_PROMPT,
    default_goal=("Find the nearest tree, break its logs by holding RT while "
                  "facing the trunk, then open the inventory and craft the "
                  "logs into planks."),
    artifact_prefix="minecraft",
    json_artifact_name="minecraft-gameplay-cycles.json",
    execute_move=execute_minecraft_move,
    execute_moves=execute_minecraft_moves,
    success_states={"planks_crafted"},
    terminal_flag_key="planks_crafted",
    terminal_evidence_key="planks_evidence",
    build_terminal_evidence=_build_terminal_evidence,
    fallback_moves=_fallback_moves,
    forced_moves=_forced_moves,
    update_stuck_counters=_update_stuck_counters,
    stuck_limits=_stuck_limits,
    stuck_message=_stuck_message,
    print_extra=_print_extra,
    cycle_extra_fields=_cycle_extra_fields,
)
