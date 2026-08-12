# 09 — GIMX Session (Start / Auth / Stop)

**Script:** `Xbox-Automation-Python/gimx-session/gimx_session.py`

The one place that owns the GIMX server process, the serial port, and the
Guide-button authentication. Run it **first**, in its own terminal, and leave it
running.

---

## 1. Why this is a separate file

Starting a session is a fundamentally different job from sending a button press:

| Session | Button press |
|---|---|
| Long-lived — runs for hours | Instant, stateless |
| **Holds COM8** (only one process can) | Holds nothing |
| Needs a **human** to hold GUIDE for 2s | Fully automatic |
| One per machine | Hundreds per test run |

Mixing them made the earlier code confusing and led to real bugs — at one point
the script would kill and restart sessions mid-run, losing the authentication.
Now the session is a separate, explicit thing you start once.

---

## 2. Usage — standalone (the normal way)

Open a terminal, start the session, and **leave it open**:

```bash
cd Xbox-Automation-Python/gimx-session
python gimx_session.py start
```

Then **hold the controller's GUIDE (Xbox) button for 2 seconds** when prompted.

Other commands:

```bash
python gimx_session.py status     # is a session up and reachable?
python gimx_session.py stop       # stop any running gimx.exe, free COM8
python gimx_session.py restart    # stop then start
python gimx_session.py start --no-stream   # start, don't tail the log
```

Options: `--console ps4`, `--port COM9`, `--gimx-config Other.xml`,
`--config path/to/controls.yaml`.

Exit codes make it scriptable: `status` returns **0** if reachable, **1** if not.

---

## 3. Usage — from Python

```python
import sys
sys.path.insert(0, "Xbox-Automation-Python/gimx-session")
from gimx_session import GimxSession, session_is_up, stop_all, running_gimx_pids

# Just check before doing work
if not session_is_up():
    sys.exit("Start a session first: python gimx_session.py start")

# Or manage one automatically (stops on exit, even on exception)
with GimxSession() as session:
    print(session.firmware)     # e.g. "8.0"
    print(session.baudrate)     # e.g. "2000000"
    ...                         # send events here
```

Useful members:

| Name | Meaning |
|---|---|
| `session_is_up()` | Is a server reachable over UDP? |
| `running_gimx_pids()` | PIDs of running `gimx.exe` |
| `stop_all()` | Stop them all, release the port |
| `GimxSession.build_command()` | The exact command that would run |
| `.adapter_detected` | Did GIMX report finding the adapter? |
| `.firmware`, `.baudrate` | Parsed from GIMX's output |
| `.log` | Every line GIMX printed |

`test_controller.py --check` already calls `session_is_up()` internally, so there
is only **one** definition of "is the session up?".

---

## 4. The command it builds

```
gimx.exe --type XOnePad --config XOnePadUsb.xml --nograb --timeout 5 \
         --src 127.0.0.1:51914 --port COM8
```

Every part matters:

| Part | Why |
|---|---|
| `--type XOnePad` | Which controller to emulate (from the console profile) |
| `--config XOnePadUsb.xml` | **Required** — without it GIMX forwards *nothing* |
| `--nograb` | Never capture mouse/keyboard — **freeze protection** |
| `--timeout 5` | Auto-exit after 5 idle minutes — runaway protection |
| `--src 127.0.0.1:51914` | Listen on UDP so other tools can send events |
| `--port COM8` | The serial port — **must be the LAST argument** |

Priority is also forced to **Normal** immediately after launch.

### Two rules learned the hard way

**`--config` is not optional.** A session started without it reported
`GIMX adapter detected` and `Firmware version: 8.0`, looked perfectly healthy,
and forwarded **no input at all** — not even from the physical controller. The
module warns you if a profile has no config file.

**`--port` must come last.** GIMX's docs: *"A --bdaddr, --port or --dst argument
finishes the current controller options."* Put it earlier and everything after it
is silently assigned to a second, non-existent controller.

`_selftest.py` asserts both of these, so a future edit can't quietly break them.

---

## 5. Safety

> **Do not run GIMX as administrator unless you must.**

Elevated, `gimx.exe` claims **realtime CPU priority** and grabs input devices.
During development this froze an entire PC — stuck mouse, nothing clickable.

Sessions started by this module always pass `--nograb` and `--timeout`, and pin
priority to Normal. **These protections apply only to sessions this module
starts** — a session you launch by hand as admin is not protected.

You will likely see this line, and it is **good news**:

```
Highest priority class can't be used due to missing elevation.
```

It means GIMX *could not* take realtime priority. The module prints a note
explaining this, because misreading it as "needs admin" is exactly what caused
the freeze.

---

## 6. What the output means

```
Starting GIMX session:
  C:\Program Files\GIMX\gimx.exe --type XOnePad --config XOnePadUsb.xml ...

  [safety] priority pinned to Normal, --nograb enabled
  [gimx] GIMX adapter detected, controller type is: XOnePad.
  [gimx] Firmware version: 8.0
  [gimx] Using baudrate: 2000000 bps.
  [gimx] found pass-through device 0x045e:0x02d1
  [gimx] Press the guide button of the controller for 2 seconds.

  adapter firmware : 8.0
  baudrate         : 2000000 bps
  listening on UDP : 127.0.0.1:51914
  serial port      : COM8

==================================================================
  ACTION REQUIRED:  hold the controller's GUIDE button
                    (the Xbox logo) for 2 SECONDS
==================================================================
```

`found pass-through device 0x045e:0x02d1` is your real Xbox controller — GIMX
uses it for the console's security handshake, which is why the Guide press is
required.

---

## 7. Important limitation

> `status` tells you the server is **reachable**. It does *not* tell you the
> session was **authenticated**.

There is no programmatic way (that we found) to confirm the Guide handshake
completed. So:

- `status` returning 0 means "a server is listening"
- It does **not** mean input will reach the console

This is the same theme as everywhere else in this project: a green result from
the sending side is not proof of effect. Only the TV is.

---

## 8. Typical workflow

```
Terminal 1                              Terminal 2
----------                              ----------
python gimx_session.py start
  (hold GUIDE 2 seconds)
  ...session stays running...
                                        python ../test-controller/test_controller.py --check
                                        python ../test-controller/test_controller.py press down*3 a
                                        python ../test-controller/test_controller.py macro nav_test
Ctrl+C to stop
```

---

## 9. Troubleshooting

| Symptom | Fix |
|---|---|
| `A GIMX session is ALREADY reachable` | One is running. `status` or `stop` first |
| `GIMX exited during startup` | Port held by another process, or the port doesn't exist |
| `Could NOT stop pid …` | That process is **elevated**; close it from its own window or an admin Task Manager |
| `WARNING: no config_file for this console profile` | That profile has no `.xml` yet — GIMX will forward nothing |
| Session up but console ignores input | Guide button almost certainly not held for 2s |

---

## 10. Offline verification

```bash
python Xbox-Automation-Python/gimx-session/_selftest.py
```

Prints the resolved settings and the exact command, and asserts the safety flags
and argument order are correct. Starts nothing and sends nothing.

---

**Related:** [05 — Test Controller](05-test-controller-docs.md) ·
[06 — Troubleshooting](06-troubleshooting.md) ·
[00 — Start Here](00-README-START-HERE.md)
