"""System prompts and schemas for the Autonomous Vision-LLM Gameplay Agent."""
from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field


class GameplayAction(BaseModel):
    """An individual controller action in an action sequence."""
    action: Literal[
        "move",
        "jump",
        "running_jump",
        "edge_jump_grab",
        "climb_or_pull_up",
        "swing_and_jump",
        "magic_marker",
        "destroy_drawing",
        "push_pull",
        "interact",
        "press_button",
        "wait",
    ] = Field(description="The macro action to execute.")
    direction: Literal["left", "right", "up", "down", "none"] = Field(
        default="none", description="Direction for movement, jumps, or stick aiming."
    )
    duration: float = Field(
        default=0.8,
        ge=0.1,
        le=4.0,
        description="Duration in seconds to hold the stick or action.",
    )
    button: str = Field(
        default="",
        description="Optional button name (e.g. 'a', 'b', 'x', 'y') if using press_button.",
    )
    run_before_jump: float = Field(
        default=0.4,
        ge=0.05,
        le=2.0,
        description="For running_jump and edge_jump_grab: run time in seconds to reach the edge.",
    )
    air_time: float = Field(
        default=0.7,
        ge=0.1,
        le=2.0,
        description="Air travel time with stick held forward toward target ledge/object.",
    )


class GameStateAnalysis(BaseModel):
    """Vision LLM reasoning output for a gameplay frame."""
    scene_state: Literal[
        "in_gameplay",
        "cutscene_or_loading",
        "death_or_respawn",
        "menu_or_prompt",
        "level_complete",
    ] = Field(description="High-level classification of the current screen.")

    player_detected: bool = Field(
        description="True if Max (blue hoodie, baseball cap, orange hair) or player character is visibly identified."
    )
    player_location: str = Field(
        description="Where Max is on screen (e.g. 'standing at edge of left stone platform', 'hanging from ledge', 'not visible')."
    )
    hazards_observed: str = Field(
        description="Visible traps, chasms, spikes, falling rocks, or pursuing creatures."
    )
    interactive_elements: str = Field(
        description="Ledges to grab, climbable vines/ropes, levers, pushable blocks, or glowing Magic Marker nodes (earth/branch/water)."
    )
    tactical_reasoning: str = Field(
        description="Step-by-step thinking: What is the immediate goal, how to navigate obstacles, and why this action was selected."
    )
    stuck_recovery_tactic: str = Field(
        default="",
        description="If previous attempt failed or Max died, explain what alternative tactic is being used to overcome the obstacle.",
    )
    actions: list[GameplayAction] = Field(
        default_factory=list,
        description="Ordered sequence of macro actions to execute for this turn.",
    )


MAX_GAMEPLAY_SYSTEM_PROMPT = """You are an Autonomous AI Game Player directly controlling the Xbox console via controller emulation.
You are playing "Max: The Curse of Brotherhood", a 2.5D cinematic puzzle-platformer.

### Core Mechanics & Controls:
1. Platforming & Ledge Grabs:
   - Max moves with the Left Stick (primarily moving right to progress).
   - Standard Jump: Press 'A'.
   - Edge Jump & Grip (`edge_jump_grab`): CRUCIAL FOR GAPS. Max runs all the way to the brink of the platform, executes a high leap (holding 'A'), and stretches arms forward-up with the stick to catch the edge of the opposite ledge or rope, pulling himself up onto the surface.
   - Climbing & Pulling Up (`climb_or_pull_up`): When hanging from a ledge or climbing a rope/vine, holds stick UP + presses 'A' to pull Max up onto the platform.
   - Rope/Vine Swinging (`swing_and_jump`): When gripping a rope or vine, builds swinging momentum left-and-right, then jumps forward at the peak of the swing.

2. Magic Marker Mechanics (Crucial for Crossing Gaps):
   - When a gap is too wide or a ledge is too high to jump across, DO NOT KEEP JUMPING TO YOUR DEATH!
   - Look around for glowing Magic Marker nodes:
     * Orange glowing earth nodes (mounds in the dirt/rock): Use `magic_marker` (direction="up") to raise an earth pillar that lifts Max or acts as a stepping stone.
     * Green glowing tree nodes: Use `magic_marker` (direction="right" or "up") to grow a branch or swingable vine.
     * Water nodes: Use `magic_marker` to spray a jet of water.
   - Reset drawing: If a pillar is misplaced or in the way, use `destroy_drawing` (presses 'X').

3. Pushing & Pulling Objects (`push_pull`):
   - Large stone blocks, carts, or fallen trees can be gripped with 'B' and pushed/pulled to bridge gaps or provide elevation.

4. Death, Respawn, and Cutscenes:
   - Touching spikes, thorns, or falling into pits results in instant death. Press 'A' immediately to respawn at the checkpoint.
   - For cutscenes, dialogs, or skip prompts (indicated by 'Y' or 'A' icons on screen), press 'A' or 'Y' to advance.

### Solving In-Game Scenarios & Stuck Obstacles:
If you are unable to cross an obstacle or died on the previous attempt:
1. DO NOT repeat the exact same failed action.
2. If Max fell into the gap:
   - Try `edge_jump_grab` with longer run_before_jump (0.5s - 0.8s) to jump from the absolute brink.
   - Check if there is an earth node underneath or in front. Use `magic_marker` to raise an earth pillar!
   - Check if there is a hanging rope/vine above the chasm. Target the rope to grip it!
3. If Max is hanging from an edge:
   - Execute `climb_or_pull_up` immediately.
4. If there is an environmental object (rock, tree branch, cart):
   - Try `push_pull` or `interact`.

Provide 1-3 macro actions per turn with precise timings (0.5s - 1.5s). Always prioritize staying alive while advancing right!
"""
