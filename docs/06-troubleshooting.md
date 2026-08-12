# 06 — Troubleshooting

Every problem we actually hit, with the real error text and the real fix. If
something breaks, search this page first.

---

## Quick diagnostic order

Work down this list — it goes from most to least common cause:

```
1. Is GIMX running?              Get-Process gimx
2. Is UDP 51914 listening?       Get-NetUDPEndpoint -LocalPort 51914
3. Did you hold GUIDE for 2s?    <- by far the most common cause
4. Was --config passed?          <- second most common
5. Can the script reach GIMX?    test_controller.py --check
6. Is COM8 free?                 see §3 below
7. Is the physical link alive?   console_connector.py
```

---

## 1. "Everything says ok but nothing happens on the console"

**The most common problem, and the most misleading.**

```
  -> right (right)   ok (remote GIMX)
  -> a (cross)       ok (remote GIMX)
```
…and the TV doesn't move.

`ok` only means GIMX accepted the event. Causes, in order of likelihood:

| Cause | Fix |
|---|---|
| **Session not authenticated** | Hold the controller's GUIDE button for **2 seconds** |
| **`--config` missing** | Restart GIMX with `--config XOnePadUsb.xml` |
| Leonardo plugged into PC, not console | Its USB must go to the console |
| Console asleep / not on a menu | Wake it, open a menu with a visible cursor |

**How to isolate it:** pick up the *physical* controller and press a button. If
the real controller doesn't work either, the problem is the **GIMX session**, not
your script. That one question saved a lot of wasted debugging.

---

## 2. PC completely froze (mouse stuck, nothing clickable)

**What happened:** GIMX was started as administrator.

```
Highest priority class can't be used due to missing elevation.
```

That message is easy to misread as "you need elevation." It actually means:
elevation lets GIMX claim **realtime CPU priority**. Elevated, it takes that
priority *and* grabs input devices — starving Windows' own input and GUI threads.
Result: total freeze.

**Prevention:**

- **Run GIMX as a normal user.** Elevation is not needed for the UDP workflow.
- If you must elevate, pass `--nograb` and a `--timeout`.
- Sessions started by our script already add `--nograb`, `--timeout 5`, and force
  Normal priority.

**Recovery:** Ctrl+Alt+Del usually still responds. Otherwise hold the power
button. Nothing is damaged — the flash and hardware are unaffected.

> Note: our safety flags only apply to sessions **our script** starts. A session
> you launch manually as admin is not protected.

---

## 3. "Access is denied" on COM8

```
async_open_path: CreateFile failed with error: Access is denied.
Error: failed to open the GIMX adapter
```

Only one process can hold a serial port. Something already has it.

```powershell
# Who's running?
Get-Process gimx, gimx-launcher -ErrorAction SilentlyContinue

# Is the port grabbable?
powershell -Command "try { $sp = New-Object System.IO.Ports.SerialPort 'COM8',500000,'None',8,'One'; $sp.Open(); 'COM8 free'; $sp.Close() } catch { $_.Exception.Message }"
```

Fix: close the other GIMX instance or the launcher session.

**If `taskkill` says "Access is denied":** that process is elevated and your
shell isn't. Close it from its own window, or use an admin Task Manager.

---

## 4. Flashing: "programmer did not respond to command: select device"

```
Found programmer: Id = "CATERIN"; type = S
avrdude.exe: error: programmer did not respond to command: select device
```

avrdude found the bootloader but the **~8-second window expired**. This is
timing, not configuration.

Fix: press RESET on the Leonardo and rerun immediately. `flash_leonardo.ps1`
polls every 200 ms specifically to win this race — press RESET *while* it says
it's waiting.

---

## 5. Flashing: "TIMED OUT: no bootloader port detected"

Checklist:

- Leonardo's USB is in the **PC**, not the console
- The cable carries **data**, not just power (many phone cables are charge-only)
- You pressed **RESET** during the waiting window
- Try a longer window: `-TimeoutSeconds 120`

---

## 6. "Bad axis name" / "Bad button"

```
Error: Bad axis name for event: lstick right(128)
```

You used a name GIMX doesn't accept. Common cases:

| Wrong | Right |
|---|---|
| `lstick right` | `lstick x` with value `127` |
| `guide` | `PS` |
| `LB` / `RB` | `l1` / `r1` |
| `LT` / `RT` | `l2` / `r2` |
| `menu` / `view` | `start` / `select` |

Full table in doc 04. Verify a new name before adding it to the config:

```powershell
& 'C:\Program Files\GIMX\gimx.exe' --type XOnePad --event 'NAME(1)' --dst 127.0.0.1:59999
```

Port 59999 is deliberately dead — GIMX still validates the name locally, so this
tests the name without sending anything.

---

## 7. "can't get controller type from remote gimx"

```
Error: can't get controller type from remote gimx
```

The client couldn't reach the server. Either the server isn't running, or it's
still starting up.

Our script retries 6 times with 1.5 s gaps, which handles the startup case. If it
persists:

```powershell
Get-NetUDPEndpoint -LocalPort 51914
```

No output = the server isn't listening. Make sure it was started with
`--src 127.0.0.1:51914`.

---

## 8. Devices that appear in Device Manager but aren't connected

Our first port scan listed an Arduino Due on COM6/COM7 and a CH340 on COM5 —
**none of them plugged in.** Windows remembers every device ever attached.

Always filter to present devices:

```powershell
Get-PnpDevice -PresentOnly -Class Ports | Select-Object Status,FriendlyName
```

Or simply:

```powershell
powershell -Command "[System.IO.Ports.SerialPort]::GetPortNames()"
```

---

## 9. The Leonardo vanished from COM ports

**This is correct behaviour after a successful flash.** With GIMX firmware the
board presents itself as a *gamepad*, not a serial port. It only appears as a COM
port while in bootloader mode.

Confirm it's alive by starting GIMX — it will report:

```
GIMX adapter detected, controller type is: XOnePad.
Firmware version: 8.0
```

---

## 10. Repeats get dropped (asked for 3, cursor moved 1)

Console menus animate, and presses arriving mid-animation can be swallowed.

Slow the repeats:

```bash
python test_controller.py press down*3 --interval 0.5
```

Or in a macro:

```yaml
- { button: down, times: 3, interval: 0.5 }
```

You can also raise `gap_between_presses` in `controls.yaml`. Our defaults are
estimates and may need tuning for your dashboard version.

---

## 11. `console_connector.py` reports silence at every baud

**Usually not a fault.** The EMUXONE firmware only answers GIMX's own handshake,
so silence is expected even on a perfectly working setup.

It *is* meaningful if GIMX also can't see the adapter. Then check:

- TX↔RX crossed correctly (FTDI TX → Leonardo RX, FTDI RX → Leonardo TX)
- **GND connected** — serial cannot work without a shared ground
- The Leonardo is powered

---

## 12. `ModuleNotFoundError: No module named 'test_controller'`

Python can't find the module. Either use the full path:

```bash
python "C:\...\Xbox-Automation-Python\test-controller\test_controller.py" --list
```

or add the directory to `sys.path`:

```python
sys.path.insert(0, "Xbox-Automation-Python/test-controller")
from test_controller import ConsolePad
```

> Note: the **folders** use hyphens (`test-controller`) but the **files** use
> underscores (`test_controller.py`) on purpose. `import test-controller` is a
> Python syntax error — the hyphen parses as subtraction.

---

## 13. `PyYAML is required`

```bash
pip install pyyaml
```

---

## 14. PowerShell won't run the flash script

```
File cannot be loaded because running scripts is disabled on this system.
```

```powershell
powershell -ExecutionPolicy Bypass -File .\flash_leonardo.ps1
```

This applies to that single run only.

---

## 15. A note on `cd` in one-off shells

When running commands through tooling, each command may start a fresh shell, so a
`cd` on one line won't persist to the next. If you see:

```
python: can't open file '...\test_controller.py': [Errno 2] No such file
```

use absolute paths.

---

## Still stuck?

Gather this before digging deeper:

```powershell
# 1. What's running
Get-Process gimx, gimx-launcher -ErrorAction SilentlyContinue

# 2. UDP listener
Get-NetUDPEndpoint -LocalPort 51914 -ErrorAction SilentlyContinue

# 3. Present COM ports
powershell -Command "[System.IO.Ports.SerialPort]::GetPortNames()"

# 4. Relevant USB devices
Get-PnpDevice -PresentOnly | Where-Object { $_.InstanceId -match 'VID_(0403|2341|045E)' } |
    Select-Object Status,Class,FriendlyName

# 5. Can the script reach GIMX
python test_controller.py --check
```

---

**Next:** [07 — Lessons Learned](07-lessons-learned.md)
