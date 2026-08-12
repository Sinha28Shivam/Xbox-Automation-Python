# 05 — Test Controller (The Main Script)

**Script:** `Xbox-Automation-Python/test-controller/test_controller.py`

This is the tool you'll use daily. It sends button presses to the console and
works both as a command-line tool and as an importable Python library.

---

## 1. Setup: start a GIMX session first

This script **only sends input**. It does not own the hardware. Starting the
session is a separate job with its own module — see
[09 — GIMX Session](09-gimx-session-docs.md).

### Step 1 — start the session (Terminal 1, leave it running)

```bash
python ../gimx-session/gimx_session.py start
```

That module builds the correct command for you (including the mandatory
`--config` and the safety flags) so you don't have to remember the details.

### Step 2 — authenticate

**Hold the controller's GUIDE (Xbox) button for 2 seconds.** GIMX will prompt
for this. Skip it and events are accepted but silently go nowhere.

### Step 3 — confirm the script can reach it (Terminal 2)

```bash
python test_controller.py --check
```

Success looks like:

```
  OK - GIMX session detected and reachable.
```

---

## 2. Everyday commands

```bash
# Show every control, macro, timing and console profile
python test_controller.py --list

# Is GIMX reachable?
python test_controller.py --check

# Single button
python test_controller.py press a

# Several buttons in order
python test_controller.py press down right a

# Repeats
python test_controller.py press down*3
python test_controller.py press down 3
python test_controller.py press down*3 right*2 a
python test_controller.py press up down --times 2
python test_controller.py press down*5 --interval 0.5

# Hold a button (long/deep press)
python test_controller.py hold guide 2.0

# Analog stick
python test_controller.py stick left_stick right --duration 1.0
python test_controller.py stick left_stick --x 64 --y -32

# Analog trigger (0-255)
python test_controller.py trigger rt 255
python test_controller.py trigger rt 128 --duration 0.5

# Named sequences from the config
python test_controller.py macro nav_test
python test_controller.py macro repeat_demo
python test_controller.py action guide_auth

# Type commands live
python test_controller.py --interactive

# Show what WOULD be sent, without sending (safe, no hardware needed)
python test_controller.py --dry-run press down*3 a
```

### Global options

| Option | Meaning |
|---|---|
| `--config PATH` | Use a different YAML config |
| `--console NAME` | Console profile (`xbox_one`, `ps4`, …) |
| `--dry-run` | Print commands, send nothing |
| `--list` | Show the control map |
| `--check` | Verify GIMX is reachable |
| `--interactive` | Live prompt |

---

## 3. Interactive mode

```
pad> a                  press A
pad> down*3             press Down 3 times
pad> down 3             same thing
pad> down*2 right a     several in one line
pad> hold guide 2       hold Guide for 2s
pad> stick left right   move left stick right
pad> trigger rt 255     pull right trigger
pad> macro nav_test     run a macro
pad> q                  quit
```

---

## 4. Using it as a Python library

```python
import sys
sys.path.insert(0, "Xbox-Automation-Python/test-controller")
from test_controller import ConsolePad

pad = ConsolePad()                          # loads config/controls.yaml

pad.press("a")                              # single press
pad.press_times("down", 3)                  # 3 times
pad.press_times("right", 5, interval=0.4)   # slower, for animated menus
pad.tap("b")                                # quick tap
pad.long_press("guide")                     # 1.0s
pad.deep_press("guide")                     # 2.0s
pad.hold("a", 0.75)                         # explicit duration

pad.stick("left_stick", "right", duration=1.0)
pad.stick("left_stick", x=64, y=-32)
pad.trigger("rt", 200, duration=0.5)

pad.run_macro("repeat_demo")
pad.run_special("guide_auth")

pad.sequence([
    {"button": "down", "times": 3, "interval": 0.3},
    {"wait": 1.0},
    {"button": "a"},
])

# Every action is recorded for later reporting
for entry in pad.action_log:
    print(entry)   # {'time':…, 'name':'a', 'gimx':'cross', 'value':1, 'status':'sent'}
```

Every method returns `True`/`False`, and `pad.failed` is set if anything failed —
useful for test assertions.

---

## 5. Understanding the output

```
  -> a (cross) 0.15s   ok
     ^   ^      ^      ^
     |   |      |      +-- GIMX accepted it
     |   |      +--------- hold duration
     |   +---------------- the GIMX name (see doc 04)
     +-------------------- your friendly name
```

With repeats:

```
  [1/3]   -> down (down) 0.15s   ok
  [2/3]   -> down (down) 0.15s   ok
  [3/3]   -> down (down) 0.15s   ok
```

### What `ok` really means

> **`ok` = "GIMX accepted this event." It does NOT mean the console reacted.**

This distinction is the single most important thing to understand about this
tool. An event can be accepted and still achieve nothing — most commonly because
the session wasn't authenticated with the Guide button.

During development the script printed `ok` six times in a row while the TV never
moved. There was no bug in the sending code; the session simply wasn't
forwarding. **Until we add screen capture (doc 08), your eyes on the TV are the
only real verification.**

---

## 6. Architecture

```
ControlConfig        loads controls.yaml, resolves aliases -> GIMX names
ConsolePad           the API: press / hold / stick / trigger / macro
  _send_event()      runs one gimx.exe client, with retries
  press()            press + ALWAYS release
  press_times()      repeat N times
  parse_repeat()     "down*3" -> ("down", 3)
  sequence()         run a list of steps
check_session()      is a GIMX server reachable?
```

### Two design decisions worth knowing

**Every press is paired with a release.** `press()` sends `cross(1)` then
`cross(0)`. If it only sent the press, the button would stay stuck down forever
and the console would become unusable.

**Failures stop the sequence.** If a press fails, remaining steps are skipped
rather than fired blindly into a state we no longer understand.

---

## 7. Why one process per button press?

Each press launches a short-lived `gimx.exe` client. GIMX's help explains why:

> *"The --event argument may require running two gimx instances."*

One instance owns the serial port; the other sends the event over UDP. Ours is
the second one.

**Cost:** roughly 250 ms per press. That's fine for menu navigation but too slow
for frame-accurate input (fighting-game combos, precise timing). If we need that
later, the fix is a persistent UDP sender speaking GIMX's protocol directly.
Noted in the roadmap.

---

## 8. Troubleshooting

| Message | Meaning and fix |
|---|---|
| `NO SESSION. Start GIMX…` | No GIMX server running, or wrong UDP port |
| `can't get controller type from remote gimx` | Server still starting. The script retries 6 times automatically |
| `Bad axis name` / `Bad button` | Invalid GIMX name in the config — see doc 04 |
| `Access is denied` on COM8 | Another process holds the port. Close other GIMX instances |
| Everything says `ok` but nothing moves | **Almost always missing Guide-button auth.** Also check `--config` was passed |
| `Unknown control 'xyz'` | Not in `controls.yaml`. Run `--list` |

---

## 9. Safe experimentation

`--dry-run` needs no hardware, no GIMX, and no console. It prints the exact
commands that would be sent. Use it freely to check your syntax:

```bash
python test_controller.py --dry-run press down*3 right*2 a
```

And `_selftest.py` verifies the config and name resolution offline:

```bash
python _selftest.py     # expects: ALL CHECKS PASSED
```

---

**Next:** [06 — Troubleshooting](06-troubleshooting.md)
