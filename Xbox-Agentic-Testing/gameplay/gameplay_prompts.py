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
        le=6.0,
        description=(
            "Duration in seconds to hold the stick or action. For "
            "magic_marker this is the STROKE length: 1.5-2.0 raises an earth "
            "pillar, but a TREE BRANCH needs 3.0-5.0 so it grows heavy "
            "enough to bend and FALL. Too short leaves a stub that never "
            "drops."
        ),
    )
    settle_after_draw: float = Field(
        default=1.5,
        ge=0.0,
        le=6.0,
        description=(
            "magic_marker only: seconds to WAIT after the stroke so the "
            "drawing finishes animating. A grown branch keeps bending, snaps "
            "and falls under its own weight after you stop drawing - use "
            "2.5-4.0 when you are waiting for a branch to fall, 1.0-1.5 for "
            "an earth pillar."
        ),
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
    # ---- Magic Marker cursor aiming ---------------------------------------
    # The cursor opens near the MIDDLE OF THE SCREEN, not on the node, so it
    # must be steered before the ink is anchored. ~450 px/s at full stick.
    node_x: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description=(
            "magic_marker / destroy_drawing: horizontal direction to steer the "
            "CURSOR from screen centre onto the glowing node (or onto the "
            "drawing to erase). -1 = left, +1 = right."
        ),
    )
    node_y: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description=(
            "magic_marker / destroy_drawing: vertical cursor steering. "
            "-1 = up, +1 = DOWN. Nodes on the ground sit BELOW screen centre, "
            "so this is usually POSITIVE."
        ),
    )
    aim_time: float = Field(
        default=0.4,
        ge=0.0,
        le=2.0,
        description=(
            "Seconds to steer the cursor before drawing. The cursor moves "
            "~450 px/s, so 0.4s is about 180px. Use 0.2s for a node near the "
            "centre, 0.8s for one near a screen edge."
        ),
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
   - IF YOU CAN SEE A GLOWING EARTH NODE, DRAW ON IT. Do not walk past it and
     do not keep pushing rocks: the node is the intended solution.
     * Orange glowing earth nodes (mounds in the dirt/rock): Use `magic_marker` (direction="up") to raise an earth pillar that lifts Max or acts as a stepping stone.
     * Green glowing tree nodes: Use `magic_marker` (direction="right" or "up") to grow a branch or swingable vine.
     * Water nodes: Use `magic_marker` to spray a jet of water.
   - Reset drawing: If a pillar is misplaced or in the way, use `destroy_drawing`.

   A BRANCH TAKES TIME - KEEP DRAWING UNTIL IT FALLS:
   A tree branch is not finished when it appears. Keep the stroke going until
   the branch is long and heavy enough to bend over and FALL - the falling
   branch is what bridges the gap, forms a ramp, or knocks an obstacle down.
     * earth pillar : duration 1.5-2.0, settle_after_draw 1.0-1.5
     * TREE BRANCH  : duration 3.0-5.0, settle_after_draw 2.5-4.0
   If a branch draw changed nothing, your stroke was TOO SHORT - raise
   `duration` and draw on the same node again. Do not abandon the node, and
   do not switch to pushing rocks.

   THE GAME TELLS YOU WHEN TO DRAW - LOOK FOR THE CONTROLLER HINT ICON:
   Max shows a small black CONTROLLER ICON on screen with one button
   highlighted in orange when it wants you to use that control here. If you
   can see a controller icon with "RT" (or a highlighted right trigger), the
   game is telling you: OPEN THE MAGIC MARKER NOW, at this spot. Treat that
   icon as a direct instruction and return `magic_marker` - do not wander off
   looking for something else, and do not report "no nodes visible" and walk
   away. The drawable node is near Max, even if the glow is subtle.
   Similarly an "A" hint means jump/confirm, "B" means grab/interact, and an
   "X" hint next to one of YOUR drawings means you can erase it.

   AIM THE CURSOR OR THE INK LANDS IN MID-AIR:
   The marker cursor opens near the MIDDLE OF THE SCREEN - not on the node and
   not on Max. Set (node_x, node_y) to steer it onto the node, comparing the
   node's position with the CENTRE of the image:
     node below-and-right of centre -> node_x=+0.7, node_y=+0.7
     node below-and-left  of centre -> node_x=-0.7, node_y=+0.7
     node straight below centre     -> node_x= 0.0, node_y=+1.0
   node_y is POSITIVE for DOWN, and ground nodes are usually below centre, so
   node_y is usually positive. The cursor moves ~450 px/s: aim_time 0.4 is
   ~180px, use 0.2 near the centre and 0.8 near a screen edge.

   STANDING ON YOUR OWN PILLAR (the key technique):
   A pillar grows UPWARD from the ground. To ride one up to a high ledge, Max
   must ALREADY BE STANDING ON the node when it grows:
     1. `move` so Max is standing directly ON the glowing mound.
     2. `magic_marker` direction="up" - the pillar carries Max up with it.
     3. `move`/`jump` onto the ledge you were trying to reach.
   If you instead need a STEPPING STONE, stand BESIDE the node, grow the
   pillar next to Max, then jump onto its top. Say which one you are doing.

   THE 'X' BADGE ON A PILLAR DOES **NOT** MEAN "DESTROY IT":
   Every drawing you make shows a small blue 'X' badge. That only means "this
   is erasable" - it is NOT an instruction. A pillar you just drew is almost
   always the SOLUTION, not an obstacle: it is there to be CLIMBED.
   Only destroy a drawing when it is genuinely sealing off the route AND you
   have already tried climbing it. If two destroy attempts in a row changed
   nothing, STOP destroying - climb the pillar instead.

   HOW TO GET ON TOP OF A PILLAR YOU DREW (do this, do not destroy it):
   A pillar is tall, so walking into its side does nothing - Max just bumps
   into it. You must approach it and jump onto its top:
     1. Note which SIDE of Max the pillar is on. If the pillar is BEHIND Max
        (to his left while he faces right), you must first `move` LEFT toward
        it - continuing to move right walks away from it forever.
     2. `move` toward the pillar to reach the base/edge next to it.
     3. `edge_jump_grab` toward the pillar, or `running_jump` with
        run_before_jump 0.5-0.8 - a standing `jump` is usually too weak to
        clear a pillar's height. RUN then JUMP from the very edge.
     4. Once on top, `move` on in your travel direction.
   If plain `jump` produced AMBIENT ONLY twice, the pillar is too high for a
   standing jump: use running_jump/edge_jump_grab, not another jump.

   IF THE LEDGE IS FAR TOO HIGH TO JUMP AT ALL - RIDE THE PILLAR UP:
   When a ledge is far above Max and no jump can reach it, do NOT jump. Stand
   ON the glowing node and grow the pillar UNDER YOURSELF so it carries you:
     1. `move` until Max is standing directly ON the glowing mound.
     2. `magic_marker` direction="up" - the pillar lifts Max as it rises.
     3. `move`/`jump` off the top onto the high ledge.
   You can also do this while ALREADY STANDING ON a pillar you drew: draw
   again on the node under you to go even higher, stacking your way up to a
   ledge that a single pillar could not reach.

   IF YOUR OWN PILLAR NOW BLOCKS THE WAY:
   A pillar you drew can wall off the route. If Max cannot get past something
   you created, use `destroy_drawing` with (node_x, node_y) aimed AT THE
   PILLAR, then move on. Erasing needs the cursor ON the pillar - there is no
   auto-snap for erasing, so aim it properly.

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

### PROGRESS IS NOT ALWAYS RIGHTWARD
Advancing usually means going right, but "right" is not a rule you must obey
when it is not working. If the thing you need - a pillar you drew, a glowing
node, a ledge - is on Max's LEFT, then move LEFT to reach it. Walking right
past your own pillar means you will never get on top of it.
Before moving, state in `tactical_reasoning` WHICH SIDE of Max the target is
on, and move toward it. If several `move right` steps in a row report AMBIENT
ONLY, Max is against a wall: stop pressing right and look up/behind instead.

### Reading the delta in your action history - IMPORTANT
The `observation` for each past step carries a pixel delta AND a verdict. Trust
the verdict, not the raw number. On this capture rig an IDLE screen already
measures 2-3 because foliage sways and dust drifts, and Max walking only
reaches about 4. So "Delta 3.6 (AMBIENT ONLY ...)" means NOTHING HAPPENED -
it is NOT progress, however plausible the number looks.
If you see two AMBIENT ONLY / NOTHING CHANGED results in a row:
  - STOP repeating that action. It is not working.
  - Something is blocking Max, or the object you are pushing will not move.
  - Look for a GLOWING NODE and use `magic_marker`. That is usually the
    intended solution, and pushing rocks usually is not.
  - If you already drew something that is now in the way, `destroy_drawing`.

Provide 1-3 macro actions per turn with precise timings (0.5s - 1.5s). Always prioritize staying alive while advancing right!
"""
