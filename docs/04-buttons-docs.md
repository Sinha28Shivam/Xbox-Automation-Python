# 04 — Button Configuration

**Config file:** `Xbox-Automation-Python/config/controls.yaml`

This is the **single global source of truth** for every control, timing and
macro. Nothing is hardcoded in the Python. Change behaviour here, and every tool
picks it up.

**This is one of the two most important documents.** The naming rules below are
deeply counter-intuitive and will cost you hours if you guess.

---

## 1. The big surprise: GIMX uses PlayStation names

GIMX internally names controls using **PlayStation terminology, even for an Xbox
One pad**. So "press A" is actually `cross`.

| You want | Xbox label | GIMX name |
|---|---|---|
| A | A | `cross` |
| B | B | `circle` |
| X | X | `square` |
| Y | Y | `triangle` |
| Menu (☰) | Menu | `start` |
| View (⧉) | View/Back | `select` |
| **Guide (Xbox logo)** | Guide | **`PS`** |
| LB | Left bumper | `l1` |
| RB | Right bumper | `r1` |
| LT | Left trigger | `l2` |
| RT | Right trigger | `r2` |
| Left stick click | LSB | `l3` |
| Right stick click | RSB | `r3` |
| D-pad | ↑↓←→ | `up` `down` `left` `right` (these are normal!) |

---

## 2. Names GIMX REJECTS

**These do not work.** We verified every one empirically by sending it to
`gimx.exe` and checking for `Bad axis name` / `Bad button`:

| Rejected ❌ | Use instead ✅ |
|---|---|
| `guide` | `PS` |
| `xbox`, `home` | `PS` |
| `LB`, `RB` | `l1`, `r1` |
| `LT`, `RT` | `l2`, `r2` |
| `menu` | `start` |
| `view`, `back` | `select` |
| `share`, `options`, `touchpad`, `sync` | (no equivalent) |
| `gas`, `brake` | `l2` / `r2` |
| `lstick right`, `lstick up` | axis + value — see §4 |

> ### This mattered in practice
> Our earlier script had `guide` mapped to `"guide"`. Since GIMX **rejects**
> that name, the Guide button would have silently failed every time. We only
> caught it by testing all candidate names against GIMX rather than trusting
> that the strings inside GIMX's own DLL were all usable. Several names *exist*
> in the binary but are **not accepted** by the event parser.

**Practical rule:** never invent a control name. Add it to `controls.yaml` only
after confirming GIMX accepts it.

---

## 3. Friendly aliases

You don't have to memorise PlayStation names. The config maps friendly names for
you, so all of these work:

```bash
press a          # -> cross
press cross      # -> cross
press confirm    # -> cross
press guide      # -> PS
press xbox       # -> PS
press home       # -> PS
press u          # -> up
press dpad_up    # -> up
press lb         # -> l1
```

See everything available:

```bash
python test_controller.py --list
```

---

## 4. Sticks are AXES, not directions

This trips everyone up. `lstick right` **is rejected**. A stick is two analog
axes. On this `XOnePad` setup they range **-32768 to +32767**, where 0 is
centred.

```yaml
left_stick:
  x_axis: "lstick x"
  y_axis: "lstick y"
  min: -32768
  max: 32767
  center: 0
  deadzone: 10
  directions:
    left:  { axis: "lstick x", value: -32768 }
    right: { axis: "lstick x", value: 32767 }
    up:    { axis: "lstick y", value: -32768 }
    down:  { axis: "lstick y", value: 32767 }
```

The `directions` block is our convenience layer so you can still write:

```bash
python test_controller.py stick left_stick right
```

...and the code translates it to `lstick x(32767)`, then returns the axis to 0
afterwards. Note **up is negative** on the Y axis — screen coordinates, not maths
coordinates.

Explicit control is also available:

```bash
python test_controller.py stick left_stick --x 64 --y -32
```

For gentler movement, scale a named direction instead of using the full throw:

```bash
python test_controller.py stick left_stick right --duration 1.0 --strength 0.2
```

`--strength` ranges from `0.0` to `1.0` and is also supported in YAML macros
for named stick directions.

---

## 5. Triggers are analog

`lt` and `rt` are not on/off — they range 0–255.

```bash
python test_controller.py trigger rt 255   # fully pressed
python test_controller.py trigger rt 128   # half pressed
python test_controller.py trigger rt 50    # light touch
```

Omitting the value uses `default_press: 255` from the config.

---

## 6. Timing

All durations live in one place, in seconds:

```yaml
timing:
  tap_duration: 0.08          # quick tap
  press_duration: 0.15        # standard press
  long_press_duration: 1.0    # long press
  deep_press_duration: 2.0    # deep press (e.g. Guide power menu)
  gap_between_presses: 0.25   # pause after each press
  menu_transition_wait: 1.0   # let a menu animate
  screen_load_wait: 3.0       # page change
  game_launch_wait: 30.0      # cold game start
  console_boot_wait: 60.0
```

Tune here rather than sprinkling `sleep()` through code.

> **Honest note:** these numbers are *reasonable starting points*, not measured
> values. `game_launch_wait: 30` in particular is a guess — real launch times
> vary hugely per title. Expect to calibrate these once we can see the screen
> (doc 08).

---

## 7. Repeating a button (2x, 3x…)

Four equivalent ways to press Down three times:

```bash
python test_controller.py press down*3      # multiply syntax
python test_controller.py press down 3      # number as its own token
python test_controller.py press down down down
```
```python
pad.press_times("down", 3)                  # from Python
```

Also `down x3`, `down:3` and `3*down` all parse. Mix freely:

```bash
python test_controller.py press down*3 right*2 a
python test_controller.py press up down --times 2   # 2x for every button listed
```

In YAML:

```yaml
- { button: down, times: 3, interval: 0.4 }
```

### Why `interval` exists
Console menus animate between rows. Presses sent too fast **can get swallowed**,
so you ask for 3 downs and the cursor moves only 1 or 2. `interval` slows just
the repeats without slowing everything else.

The defaults (0.3–0.8s) are educated guesses. If repeats get dropped, raise it.

---

## 8. Macros

Reusable named sequences:

```yaml
macros:
  repeat_demo:
    description: "down x3, right x2, then confirm"
    steps:
      - { button: "down",  times: 3, interval: 0.3 }
      - { button: "right", times: 2, interval: 0.3 }
      - { wait: 0.5 }
      - { button: "a" }
```

Run it:

```bash
python test_controller.py macro repeat_demo
```

Step keys: `button`, `times`, `interval`, `duration`, `wait`, `trigger`+`value`,
`stick`+`direction`/`x`/`y`, and `strength` for named stick directions.

Example:

```yaml
- { stick: left_stick, direction: left, duration: 0.5, strength: 0.2 }
```

---

## 9. Special actions and the `verified` flag

```yaml
special_actions:
  guide_auth:
    description: "Hold Guide 2s - GIMX session authentication"
    verified: true          # <-- confirmed on hardware

  screenshot:
    description: "Xbox screenshot"
    verified: false         # <-- BEST GUESS, not confirmed
```

**`verified: false` means we have not proven this sequence works.** The button
combos for screenshot, clip recording and the power menu are based on Xbox
conventions but were never tested on the console. The code prints a warning when
you run an unverified action.

Please flip these to `true` only after seeing them work on the TV — and that
honesty is deliberate, so nobody inherits a false assumption.

---

## 10. Console profiles

```yaml
consoles:
  xbox_one:
    gimx_type: "XOnePad"
    firmware: "EMUXONE"
    config_file: "XOnePadUsb.xml"
    default: true
  ps4:
    gimx_type: "DS4"
    firmware: "EMUPS4"
```

Switching console = flash the matching firmware (doc 02) and run with
`--console ps4`. Only `xbox_one` is hardware-tested; the others are configured
from GIMX's documented types but unproven here.

---

## 11. Offline verification

Check the config parses and names resolve without any hardware:

```bash
python temp_code_test/hardware/test_controller_selftest.py
```

Expected: `ALL CHECKS PASSED`.

---

**Next:** [05 — Test Controller](05-test-controller-docs.md)
