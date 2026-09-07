# Magic Marker — hardware-verified findings

## ✅ BOTH OPERATIONS CONFIRMED WORKING

**`--aim-time 0.4` with `--strength 1.0`, on `lstick`, for both modes.**

```
# draw a pillar
python _manual_marker.py --aim down-right --aim-time 0.4 --draw up \
    --draw-time 1.8 --strength 1.0

# destroy a pillar
python _manual_marker.py --mode destroy --aim down-right --aim-time 0.4 \
    --strength 1.0
```

Both user-confirmed on hardware. Destroy verified in the frames too:
`4destroying` shows the cursor on the pillar mid-erase, and `6after` shows a
**clean scene with the pillar gone**.

### Measured cursor gain (the number that was missing all along)

`2amidaim` is captured at `aim_time * 0.5`, so it gives a direct read:

| run | travel | elapsed | gain |
|---|---|---|---|
| draw | 82.6 px | 0.20 s | **~413 px/s** |
| destroy | 101.6 px | 0.20 s | **~508 px/s** |

So the cursor moves roughly **400-500 px/s at full deflection**, i.e. about
**80-100 px per 0.2 s**. To move N px, hold ~`N/450` seconds.

This finally explains the whole saga:

- `0.4 s` at strength 1.0 -> ~180 px of travel, which reaches the target.
- `0.4 s` at strength **0.4** -> only 6.7 px. Deflection scales the speed
  hard, so a reduced strength effectively freezes the cursor. The earlier
  failed destroy run had `strength 0.4` entered at the strength prompt.
- `0.1 s` -> ~40 px. That was enough for *drawing*, because the ink snaps to
  nearby glowing earth, but far too little for *destroying*, which has no
  snap target and needs the cursor physically on the pillar.

### `aimed` reporting "not found" is expected

In the successful destroy run the reticle was `not found` at the `2aimed`
stage. That is fine: the cursor sits over the dark pillar/rock, so it stops
matching the pale-disc rule. Movement is measured from `open` -> `mid-aim`,
which is why mid-aim capture was added. Do not "fix" this.

## Stick choice differs between DRAW and DESTROY

**Drawing uses `lstick` and is confirmed working — do not change it.**

I briefly switched the default to `rstick` after measuring only a 9 px reticle
shift during the aim stage (`1open` (967.3, 533.6) -> `2amidaim` (959.4,
537.9), `dx=-8.0 dy=+4.3`). **That change was wrong and has been reverted.**
Drawing demonstrably produces pillars on `lstick`, so the ink anchor does not
depend on that measured shift — the reticle metric is simply a poor proxy for
where the stroke lands. Hardware-observed success outranks the pixel metric.

`rstick` was then tried for **destroy** mode and **also did not work**, so the
stick was never the variable that mattered. Both modes now use `lstick`, and
destroy reuses the proven draw aiming verbatim: `lstick`, `--aim-time 0.1`,
`--strength 1.0`.

| mode | `--aim-stick` | `--aim-time` |
|---|---|---|
| `draw` | `lstick` | 0.1 |
| `destroy` | `lstick` | 0.1 |

`--draw-stick` also defaults to `lstick`. Every stage zeroes **all four**
stick axes so a stale deflection cannot leak between phases.

### Destroy is still unconfirmed — likely not an aiming problem

Since neither stick nor aim-time changed the outcome, the remaining suspects
are about *when* X is sent, not *where* the cursor is:

- **The cursor re-centres before X lands.** The aim stage releases the stick
  and waits 0.4 s before the X press, so the cursor may drift back to rest.
  `--hold-aim` keeps the stick deflected through the X press.
- **One tap is not enough** -> `--repeat-x 3`.
- **The tap is too short** -> `--x-hold 0.6`.
- **The aim direction is stale.** A successful draw *pans the camera*, so the
  pillar is no longer where the glowing earth was. `down-right` is correct for
  the glow, not necessarily for the finished pillar.
- **X may not be the erase button at all.** The X-prompt beside the pillar is
  suggestive but not proof; it could be a *context* action (climb/interact).
  If the escalation above all fails, the binding itself needs re-testing.

## Destroy / erase a drawing

X (gimx **`square`**, per `controls.yaml`) erases an existing drawing. The
X-prompt icon sitting next to the finished pillar in `_mm_1downright_4drawing`
and `_6after` is the in-game confirmation of this binding.

```
python _manual_marker.py --mode destroy --aim down-right
```

Escalate if X erases nothing:

```
python _manual_marker.py --mode destroy --aim down-right --hold-aim
python _manual_marker.py --mode destroy --aim down-right --hold-aim --repeat-x 3 --x-hold 0.6
```

Sequence: hold RT -> aim at the **pillar** -> tap X -> release RT. Two things
matter:

- **RT stays held through the X press.** X only erases while the marker is
  open; releasing RT first just closes the marker.
- **Aim at the pillar, not the glow.** Once drawn, the pillar occupies a
  different screen position than the glowing earth it grew from - and the
  camera pans after a successful draw, so the old aim direction is stale.

`--repeat-x N` taps X several times for drawings that need more than one hit.
`square` is now also cleared in the baseline and in the `finally` block, so a
Ctrl-C can't leave X stuck down.

### Success vs failure, numerically

| measure | down-right (worked) | up-right (failed) |
|---|---|---|
| persistent changed area (`0closed`→`6after`) | **1 180 442** | 46 914 |
| largest persistent blob | **1 035 925** (full-frame: camera panned) | 18 351 |

A successful pillar changes the frame by an order of magnitude more, because
the **camera pans** to follow the new structure. That whole-frame shift is
itself a strong success signal, though it also means "biggest blob" is useless
for locating the pillar afterwards — the whole view moved.



Evidence: `_mm_1upright_*.png` (a real `_manual_marker.py --aim up-right` run,
1920x1080) plus a user-supplied reference screenshot.

## 1. RT range — FIXED

On XOnePad the trigger axis spans **0..32767**, like a stick, **not 0..255**.

| `r2` value | Prompt-ROI delta | Marker |
|---|---|---|
| 255 | 0.2–0.3 | does not open |
| 512 | 0.2–0.3 | does not open |
| 1023 | 65.9 | **opens** |
| 32767 | 66.3 | **opens** |

`gimx.exe` *accepts* `r2(255)` and reports success, so every layer logged "ok"
while the console saw an essentially unpressed trigger. Activation threshold is
between 512 and 1023. Fixed in `controls.yaml`, `game_tools.py`, and
`gameplay_vision.py` (the last still had a stale `255` fallback).

## 2. Holds must be atomic — FIXED

Each `pad._send_event()` spawns its own `gimx.exe` (~250 ms) and carries a
**state, not a latch**. Setting RT, then A, then the stick sequentially meant RT
had already lapsed before the stroke began, so the marker closed mid-draw.

`gimx.exe` accepts **multiple `--event` flags in one invocation**. Added
`ConsolePad._send_events()`; `draw_magic_marker_impl` now asserts RT+A+stick
together with keep-alive re-sends. Also fixed A being sent `255` (buttons are
`1`/`0`).

## 3. Left stick does NOT move the character in marker mode

Max stayed at **x≈1161, y≈873 (±3 px)** across all seven stages while RT was
held and the stick was pushed up-right. So the left stick drives the marker
cursor, not Max. (Earlier delta-based "he's stuck" conclusions were wrong.)

## 4. The blob at (974, 530) is the CURSOR, not glowing earth — ROOT CAUSE

The 60 px box at (974,530) measures `gray 66 / sat 123` when closed and
`gray 194 / sat 53` when open: a bright, **desaturated** disc that exists only
while RT is held. That is the pale marker reticle sitting up in the tree
canopy, which is where the cursor rests when the marker opens.

My detector looked for "regions that brighten when the marker opens" and this
reticle is the single biggest such region — so **the aim vector was computed
toward the cursor's own start position**. It could never be right. This is why
ink landed in the wrong place.

## 5. The real glowing earth

The drawable glow is the **bright amber pool on the ground just right of Max**,
found at **(1246, 891)** — `dx=+85, dy=+16` from Max, i.e. essentially
**level-right, ~86 px away**, hugging the base of the big rock.

Its signature: present in `1open`, absent in `0closed`, area ~609 px, tall-thin
box 22x63. Distinguishing it from sunlit sand needs **both** high brightness
**and** the marker-open appearance test — plain amber thresholding returns the
large sand pools at (937,931) and (719,952), which are present in *both*
states.

## 6. Detector rules that follow

- **Never** treat "brightest new region on marker open" as the target — that is
  the reticle. Exclude pale/desaturated discs (`S < 110`, `V > 170`, roughly
  circular, `fill > 0.45`).
- Require the glow to be **saturated amber** (`H 18–42`, `S > 90`, `V > 200`)
  *and* to appear only when the marker is open.
- Restrict the search to a band around Max's feet; the correct target was
  86 px away while foliage decoys sat 400+ px up-frame.
- Locate Max by **blue shirt in the lower frame with orange hair above it**.
  Unconstrained blue matching returns the sky (area 194 872 at y≈127).

## 7. Capture pipeline lag

A frame grabbed immediately after an action can still show the **previous**
screen. All diagnostics flush 2–3 frames before measuring. This once made a
working Guide press look like dead input.

## 8. Stuck-detection: whole-frame delta is unusable

Idle foliage alone gives 2–3; walking gives 2.7–4.2. The ranges overlap, so the
threshold cannot separate them. Use **position tracking** instead (Max's shirt
is highly selective on this palette).

## 9. The pale-disc reticle detector is NOT reliable — do not ship it

In the successful run it reported a **static** blob at (78, 798) with area
~26 000 in *every* stage, including `0closed`. That is scenery at the frame
edge, not the cursor. Meanwhile the visible reticle in `_pl_2aimed.jpg` sits
near the centre-right of the frame.

So "bright + desaturated + roundish" is too weak a rule: it must additionally
require the blob to be **absent when the marker is closed**. The gated version
(subtract the closed-frame pale mask first) *did* correctly isolate the reticle
at (970, 533) — that gating is mandatory, not optional.

## 10. Aiming is a TAP, not a hold

The cursor traverses the screen quickly. `--aim-time 0.6` overshot so far that
the ink landed in a different part of the scene; **0.1 s** lands on target.
Any closed-loop implementation should therefore use short pulses and re-check
the reticle between them, rather than one long computed deflection.

## 11. Outstanding

The reticle was only observed at its rest position: frames were captured
**after** the stick was re-centred, so the stick→cursor gain (px/s per unit
deflection) is still unmeasured. Calibration must sample **while the
deflection is held**.
