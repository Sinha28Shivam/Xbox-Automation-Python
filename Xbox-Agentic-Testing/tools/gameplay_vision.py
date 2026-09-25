"""gameplay_vision.py - closed-loop, vision-guided gameplay for Max: The
Curse of Brotherhood.

The game-specific logic (platforming macros, Magic Marker aim/draw/erase,
frame schema, prompt, death/draw/destroy counters) lives in
game_profiles/max_profile.py. The shared observe->decide->act->re-observe
loop, LLM plumbing and frame capture live in gameplay_engine.py. This module
keeps the historical `vision_gameplay_loop` name/signature as a thin wrapper
since tools/game_tools.py imports it directly.
"""

from __future__ import annotations

from typing import Any

from registry import ToolContext

from gameplay_engine import run_gameplay_loop
from game_profiles.max_profile import PROFILE


def vision_gameplay_loop(
    ctx: ToolContext,
    goal: str = "Advance through the level, drawing with the Magic Marker "
                "where needed, until a checkpoint is reached.",
    max_cycles: int = 40,
    cycle_delay: float = 0.4,
    stop_on_checkpoint: bool = True,
    settle_after_move: float = 0.6,
    stuck_limit: int = 6,
) -> dict[str, Any]:
    """Play the game by looking at every frame and deciding what to do.

    See gameplay_engine.run_gameplay_loop for the loop's end conditions and
    evidence shape. `stop_on_checkpoint` maps onto the engine's generic
    `stop_on_success`.
    """
    return run_gameplay_loop(
        ctx, PROFILE, goal=goal, max_cycles=max_cycles,
        cycle_delay=cycle_delay, settle_after_move=settle_after_move,
        stuck_limit=stuck_limit, stop_on_success=stop_on_checkpoint,
    )
