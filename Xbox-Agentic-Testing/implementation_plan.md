# Integrating calibrated Magic Marker into vision-guided gameplay

## What already exists (no need to rebuild)

The closed-loop infrastructure is **already complete and well-built**:

| piece | file | status |
|---|---|---|
| per-frame observe→decide→act loop | `tools/gameplay_vision.py` | ✅ works |
| vision model + structured decisions | `FrameDecision` / `GameplayMove` | ✅ works |
| checkpoint stop condition | `scene_state == checkpoint_reached` | ✅ works |
| death/cutscene handling | respawn + advance_prompt | ✅ works |
| stuck detection, per-cycle evidence | `vision_gameplay_loop` | ✅ works |
| tool exposure to the planner | `vision_guided_gameplay` | ✅ works |
| autonomous CLI entry | `console.py play` | ✅ works |

So this is **not** a new feature. It is a surgical correction of the two
marker macros, which are the only reason the loop cannot finish a level.

## Gap 1 — `draw_marker_stroke` never aims (CRITICAL)

`tools/gameplay_vision.py:320`. Current sequence:

```
hold RT -> hold A -> push stick along (aim_x, aim_y) -> release
```

**There is no aim phase.** It holds A immediately, so the ink anchors at the
cursor's *rest* position instead of on the glowing node. Hardware-measured
rest is `(967, 534)`; the real glow was at `(1246, 891)` — off by ~450 px.

It only ever appeared to work because the game **auto-snaps** ink to a nearby
node. Outside snap range it silently draws in mid-air.

Correct sequence, from `MARKER_FINDINGS.md`:

```
hold RT -> AIM: push stick toward the node 0.4s @ strength 1.0
        -> hold A (anchor) -> stroke -> release A -> release RT
```

## Gap 2 — `destroy_drawing` cannot work (CRITICAL)

`tools/gameplay_vision.py:455` is the whole implementation:

```python
pressed = pad.press("x", duration=0.20)
```

Two hardware-verified requirements are missing:

1. **RT is not held.** X only erases *while the marker is open*. Pressing X
   during normal gameplay is an unrelated context action.
2. **No aiming.** The cursor must physically sit on the drawing. There is no
   snap-to for erasing, which is exactly why destroy failed for so long while
   draw succeeded.

## Gap 3 — non-atomic holds

The macros use one `_send_event` per axis. `ConsolePad._send_events()` was
added precisely so RT+A+stick land in a single `gimx.exe` call; a multi-call
sequence can drop a hold mid-stroke.

## Calibrated constants to fold in

| constant | value | source |
|---|---|---|
| RT press | `32767` (≥1023 required) | trigger axis is 0..32767, not 0..255 |
| aim time | **0.4 s** | confirmed for draw *and* destroy |
| aim strength | **1.0** | 0.4 gives ~17 px/s — cursor effectively frozen |
| cursor gain | **~450 px/s** at full deflection | 82.6 px & 101.6 px in 0.20 s |
| stroke time | 1.8 s | proven pillar height |
| X button | gimx `square` | `controls.yaml` |

**Strength matters more than time** — the single most important lesson.

## Changes

### 1. `GameplayMove` — let the model aim
Add `node_x`, `node_y` (stick vector from Max toward the glowing node) and
`aim_time` (default 0.4). Keep `aim_x`/`aim_y` as the *stroke* vector, so the
existing prompt stays valid.

### 2. `draw_marker_stroke(...)` — insert the aim phase
Aim toward `(node_x, node_y)` at full strength for `aim_time` **before**
pressing A. Use `_send_events` for atomicity.

### 3. `destroy_marker_drawing(...)` — new function
Hold RT → aim at the drawing → tap X (`square`) → release RT. Replaces the
bare `pad.press("x")`.

### 4. Prompt update
Teach the model: aim at the node first; the cursor starts near screen centre;
full deflection ≈ 450 px/s.

## Verification

Reuse the proven signal: a real pillar changes the frame by ~1.18 M px
(vs 47 K for a failure) because the camera pans. The loop already records
per-cycle deltas, so this needs no new plumbing.

## Explicitly out of scope

- Per-frame glow detection in `game_tools.py` (the vision model already
  locates nodes; a second CV detector would just disagree with it)
- Touching the working loop, checkpoint logic, or death handling
- `_manual_marker.py`, which stays as the standalone calibration rig
