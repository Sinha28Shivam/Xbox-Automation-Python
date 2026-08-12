# Xbox Console Automation — Start Here

**New to this project? Read this page first.** It explains what we built, why it
exists, and where to go next. No prior knowledge assumed.

---

## 1. What problem does this solve?

Normally a human picks up a controller and presses buttons to navigate an Xbox.
We want a **computer program** to do that instead, so we can automatically test
consoles and games — launch a title, walk through menus, confirm it worked, and
repeat that hundreds of times without a person sitting there.

The catch: a console will not accept commands from a PC over USB. It only trusts
a **real game controller**. So we have to build something the console *believes*
is a controller, but that we can drive from Python.

---

## 2. How it works (the big picture)

We turned an Arduino Leonardo into a fake Xbox controller. Here is the full
chain, and this diagram is worth understanding before anything else:

```
   YOUR PYTHON SCRIPT
          |
          v
   gimx.exe  (CLIENT)          "press A"
          |
          |  UDP network message (127.0.0.1:51914)
          v
   gimx.exe  (SERVER)          owns the hardware
          |
          |  USB cable
          v
   FTDI UART adapter  (COM8)   USB <-> serial converter
          |
          |  TX / RX wires
          v
   ARDUINO LEONARDO            flashed with GIMX firmware
          |                    the console sees this as a controller
          |  USB cable
          v
   XBOX CONSOLE
```

**Why is the Arduino Leonardo special?** Most Arduinos (like the Uno) cannot
pretend to be a USB device such as a keyboard or gamepad. The Leonardo uses an
ATmega32U4 chip with **built-in USB support**, so it can. That is the entire
reason this board was chosen.

**What is GIMX?** Free software that speaks the console's controller protocol.
It provides the firmware for the Leonardo *and* the PC-side program that talks
to it. We do not reimplement any of this — we drive GIMX.

**Why two copies of gimx.exe?** GIMX only lets one program own the serial port.
The "server" holds the hardware; the "client" is a short-lived process that
sends one button event and exits. GIMX's own help text says: *"The --event
argument may require running two gimx instances."*

---

## 3. The documents

Read them in this order:

| # | Document | What it covers |
|---|---|---|
| 01 | [Hardware Setup](01-hardware-setup.md) | The physical parts and how they connect |
| 02 | [Flashing the Leonardo](02-leonardo-flash-docs.md) | Putting controller firmware on the Arduino |
| 03 | [Console Connector](03-console-connector-docs.md) | Diagnosing the connection when things break |
| 04 | [Button Configuration](04-buttons-docs.md) | The control map and why names are surprising |
| 05 | [Test Controller](05-test-controller-docs.md) | The main script — sending button presses |
| 06 | [Troubleshooting](06-troubleshooting.md) | Every problem we hit and how we fixed it |
| 07 | [Lessons Learned](07-lessons-learned.md) | Mistakes made during development — read this |
| 08 | [Roadmap](08-roadmap-agentic-framework.md) | The AI testing framework we're building next |

**If you only read two, read 04 (buttons) and 07 (lessons learned).** Those two
contain the non-obvious knowledge that cost the most time to discover.

---

## 4. Project layout

```
xboxArudino/
├── docs/                      <- you are here
└── Xbox-Automation-Python/
    ├── config/
    │   └── controls.yaml      <- GLOBAL config: buttons, timings, macros
    ├── console-connector/
    │   └── console_connector.py   <- connection diagnostic tool
    ├── flash-leonardo/
    │   └── flash_leonardo.ps1     <- flashes firmware onto the Arduino
    ├── test-controller/
    │   ├── test_controller.py     <- MAIN script: sends button presses
    │   └── _selftest.py           <- offline check, touches no hardware
    └── requirements.txt
```

---

## 5. Quick start

Assuming the hardware is already wired and flashed:

```bash
# 1. Install the one dependency
pip install pyyaml

# 2. See every button you can press
python Xbox-Automation-Python/test-controller/test_controller.py --list

# 3. Start GIMX (see doc 05) and hold the controller's GUIDE button for 2 seconds

# 4. Confirm the script can reach GIMX
python Xbox-Automation-Python/test-controller/test_controller.py --check

# 5. Press some buttons
python Xbox-Automation-Python/test-controller/test_controller.py press down*3 a
```

Add `--dry-run` to any command to see what *would* be sent without sending it.
This is completely safe and needs no hardware.

---

## 6. The single most important caveat

> **A command that prints `ok` does NOT prove the console did anything.**

`ok` means "GIMX accepted the event." The event can still fail to reach the
console — for example if the GIMX session was never authenticated with the Guide
button. During development this exact confusion wasted hours: the script
cheerfully reported success six times in a row while the TV screen never moved.

Until we add a capture card and screen-reading (see doc 08), **the only real
proof is looking at the television.** Please keep that in mind whenever you see
a green-looking result.

---

## 7. Safety warning

**Do not run GIMX as administrator unless you have a specific reason.** When
elevated, it claims realtime CPU priority and grabs your input devices — this
froze a PC completely during development (stuck mouse, nothing clickable).
Details in doc 06.
