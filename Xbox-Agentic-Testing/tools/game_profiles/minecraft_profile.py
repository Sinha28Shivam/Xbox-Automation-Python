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

from combo_dispatch import ComboComponent, dispatch_combo
from gameplay_engine import GameProfile
from minecraft_controls import button_for


# ===========================================================================
# What the model is allowed to decide
# ===========================================================================
class MinecraftMove(BaseModel):
    """One controller action. Several may be returned per cycle."""

    action: Literal[
        "move", "look", "mine", "attack", "jump", "sprint", "interact",
        "press_button", "navigate_recipe", "craft_all", "wait",
    ] = Field(description="The controller action to execute.")

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
        default=0.8, ge=0.1, le=6.0,
        description="Seconds to hold the stick/trigger/button. For 'mine' "
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
    ] = Field(description="Classification of the current screen.")

    player_visible: bool = Field(
        description="Is the first-person view / hotbar / crosshair visible?")
    tree_visible: bool = Field(
        description="Is a tree trunk visible anywhere in the frame?")
    log_visible: bool = Field(
        description="Is a wood log block visible directly in front of the "
                    "crosshair (close enough to mine)?")

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

While pursuing that goal, also report any mob, structure, or resource you
notice in `nearby_entities` (e.g. a zombie, a distant house, an ore vein) -
this is purely observational and does not change which moves are valid this
cycle. Only set `is_threat` true for a hostile mob close enough to matter
right now; do not flag distant or passive mobs as threats.

CONTROLS AVAILABLE TO YOU (as macros)
  move          : left stick, walk/run in `direction` (forward=up, back=down)
  look          : right stick, turn the camera in `direction` to find/face a tree
  mine          : hold RT while facing a block, breaks it after a few seconds
                  of continuous holding. Must be standing close enough that
                  the block is directly under the crosshair.
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
_SIMULTANEOUS_COMPATIBLE = {"move", "look", "attack", "mine", "jump", "sprint"}

_LOOK_STRENGTH = 0.6  # gentler than a full deflection, or the camera whips too fast to track a target


def _minecraft_move_to_component(move: MinecraftMove) -> tuple[ComboComponent, float]:
    """One MinecraftMove -> (ComboComponent, hold duration for that component)."""
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
    # No decision is still a decision: turn to look for a tree rather than
    # burning a cycle standing still.
    return [MinecraftMove(
        action="look", direction="right", duration=0.6,
        purpose="Model returned no move; look around for a tree.")]


def _print_extra(decision: MinecraftFrameDecision) -> None:
    print(f"  Tree seen : {decision.tree_visible}", flush=True)
    print(f"  Log seen  : {decision.log_visible}", flush=True)
    if decision.scene_state == "inventory_open":
        print(f"  Recipe    : {decision.recipe_highlighted or '(none)'} "
              f"x{decision.craftable_count}", flush=True)
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
        counters["mine_streak"] = 0 if progressed else counters.get("mine_streak", 0) + 1
    else:
        counters["stuck_streak"] = 0 if progressed else counters.get("stuck_streak", 0) + 1
        counters["mine_streak"] = 0


def _stuck_limits(stuck_limit: int) -> dict[str, int]:
    base = max(2, int(stuck_limit))
    return {"stuck_streak": base, "mine_streak": base * 2}


def _stuck_message(name: str, value: int) -> str:
    if name == "mine_streak":
        return (f"{value} consecutive mine attempts produced no visual "
                f"change even at a relaxed threshold. Either input is not "
                f"reaching the console or the model is mining an "
                f"unreachable/incorrect block. Stopping instead of sending "
                f"more input into a void.")
    return (f"{value} consecutive non-mining cycles produced no visual "
            f"change. Either input is not reaching the console (an "
            f"unauthenticated GIMX session is the usual cause) or the model "
            f"cannot find/reach a tree. Stopping instead of sending more "
            f"input into a void.")


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
    update_stuck_counters=_update_stuck_counters,
    stuck_limits=_stuck_limits,
    stuck_message=_stuck_message,
    print_extra=_print_extra,
    cycle_extra_fields=_cycle_extra_fields,
)
