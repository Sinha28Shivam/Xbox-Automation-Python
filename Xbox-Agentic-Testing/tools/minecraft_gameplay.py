"""minecraft_gameplay.py - tool registration for Minecraft vision gameplay.

The game-specific logic (move vocabulary, frame schema, prompt, dispatch,
mine/stuck counter split) lives in game_profiles/minecraft_profile.py. The
shared observe->decide->act->re-observe loop lives in gameplay_engine.py.
This module only wires the two together as a registered tool.
"""

from __future__ import annotations

from typing import Any

from registry import ToolContext, ToolSpec, make_tool

from gameplay_engine import run_gameplay_loop
from game_profiles.minecraft_profile import PROFILE


def vision_guided_minecraft_gameplay_impl(
    ctx: ToolContext,
    goal: str = ("Find the nearest tree, break its logs by holding RT while "
                 "facing the trunk, then open the inventory and craft the "
                 "logs into planks."),
    max_cycles: int = 45,
    # See combat_tools.engage_single_mob_impl's note: combined move+look
    # dispatch removed most of the sequential subprocess overhead this used
    # to compensate for.
    cycle_delay: float = 0.2,
) -> dict[str, Any]:
    return run_gameplay_loop(ctx, PROFILE, goal=goal, max_cycles=max_cycles,
                             cycle_delay=cycle_delay)


def _vision_guided_minecraft_gameplay(ctx: ToolContext) -> Any:
    def run(goal: str = ("Find the nearest tree, break its logs by holding "
                         "RT while facing the trunk, then open the "
                         "inventory and craft the logs into planks."),
            max_cycles: int = 45,
            cycle_delay: float = 0.2) -> dict[str, Any]:
        return vision_guided_minecraft_gameplay_impl(
            ctx, goal=goal, max_cycles=max_cycles, cycle_delay=cycle_delay)

    return make_tool(
        run, "vision_guided_minecraft_gameplay",
        "Play Minecraft closed-loop with a VISION model: every cycle it "
        "looks at the live frame and decides moves (move, look, mine, "
        "interact, press_button, wait) to find a tree, break its logs, and "
        "craft planks. Keeps playing until planks are seen, max_cycles is "
        "reached, or nothing on screen changes any more. Returns a "
        "per-cycle log of frames, deltas and reasoning as evidence.")


def provide() -> list[ToolSpec]:
    return [
        ToolSpec(name="vision_guided_minecraft_gameplay",
                 description=("Closed-loop vision gameplay for Minecraft: look at "
                              "the live frame every cycle, find a tree, break its "
                              "logs, open the inventory, and craft planks."),
                 tags=["input", "vision", "game"],
                 factory=_vision_guided_minecraft_gameplay, mutates_hardware=True),
    ]
