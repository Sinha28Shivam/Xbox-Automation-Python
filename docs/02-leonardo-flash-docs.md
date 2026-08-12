# 02 — Flashing the Arduino Leonardo

**Script:** `Xbox-Automation-Python/flash-leonardo/flash_leonardo.ps1`

"Flashing" means writing new firmware onto the Arduino's chip. We replace
whatever was on the board with GIMX firmware that makes it behave like an Xbox
controller.

---

## 1. What flashing does

The Leonardo ships as a general-purpose Arduino. We overwrite its program with
**EMUXONE.hex** — GIMX's Xbox One controller emulator. After flashing:

- The board no longer appears as an Arduino serial port
- It presents itself to the console as a **gamepad**
- It listens for instructions on its serial (TX/RX) pins

> The board *disappearing* from your COM port list after a successful flash is
> **expected behaviour**, not a failure. This surprises people.

---

## 2. Available firmware

In `C:\Program Files\GIMX\firmware\`:

| File | Target console |
|---|---|
| **EMUXONE.hex** | **Xbox One** ← what we use |
| EMU360.hex | Xbox 360 |
| EMUXBOX.hex | Original Xbox |
| EMUPS4.hex | PlayStation 4 |
| EMUPS3.hex | PlayStation 3 |
| EMUJOYSTICK.hex | Generic USB joystick (useful for PC testing) |
| EMUG27 / EMUG29PS4 / EMUDF / EMUDFP / EMUGTF | Racing wheels |

To target a different console, flash the matching firmware **and** update the
console profile in `config/controls.yaml`.

---

## 3. The 8-second problem (read before flashing)

The Leonardo has a bootloader — a small program that accepts new firmware. It
runs **only for about 8 seconds after a reset**, then hands control to the main
program. If you don't start flashing inside that window, it closes.

This caused a real failure during development. avrdude connected and identified
the bootloader:

```
Found programmer: Id = "CATERIN"; type = S
avrdude.exe: error: programmer did not respond to command: select device
```

It found the bootloader but the window had already expired mid-operation. The
fix isn't a code change — it's **timing**.

That's why `flash_leonardo.ps1` **polls for the bootloader port every 200 ms and
fires avrdude the instant it appears**, rather than making you race a stopwatch.

---

## 4. How to flash

```powershell
cd Xbox-Automation-Python\flash-leonardo
powershell -ExecutionPolicy Bypass -File .\flash_leonardo.ps1
```

Then:

1. **Connect the Leonardo's USB to the PC** (not the console — it needs to be
   flashed by the PC).
2. The script prints `Waiting up to 90 s for a bootloader COM port...`
3. **Tap the RESET button once** on the Leonardo.
4. Flashing starts automatically and takes under a second.

Options:

```powershell
# Different firmware
.\flash_leonardo.ps1 -Firmware EMUPS4

# Longer window to get ready
.\flash_leonardo.ps1 -TimeoutSeconds 120

# Skip auto-detection if you already know the port
.\flash_leonardo.ps1 -ForcePort COM9
```

`-ExecutionPolicy Bypass` is needed because Windows blocks unsigned PowerShell
scripts by default. It applies only to this one run.

---

## 5. What success looks like

```
>>> Bootloader detected on COM9 - flashing NOW <<<
Found programmer: Id = "CATERIN"; type = S
avrdude.exe: AVR device initialized and ready to accept instructions
avrdude.exe: Device signature = 0x1e9587 (probably m32u4)
avrdude.exe: writing flash (4580 bytes):
Writing | ################################################## | 100% 0.39s
avrdude.exe: 4580 bytes of flash written
avrdude.exe: verifying ...
avrdude.exe: 4580 bytes of flash verified          <-- THE KEY LINE
avrdude.exe: safemode: Fuses OK (E:CB, H:D8, L:FF)

SUCCESS: EMUXONE flashed to the Leonardo.
```

**Why this output is trustworthy.** Unlike most results in this project, this one
is genuinely proven:

- `Device signature = 0x1e9587` — the chip identified itself as an ATmega32U4,
  so we were definitely talking to the right hardware
- `4580 bytes of flash **verified**` — avrdude read the firmware **back off the
  chip** and compared it byte-for-byte with the source file

That read-back is real evidence, not an assumption. Compare this with sending
button presses (doc 05), where "ok" only means the command was accepted.

You can double-check afterwards — the adapter reports its own firmware version
when GIMX connects:

```
GIMX adapter detected, controller type is: XOnePad.
Firmware version: 8.0
```

---

## 6. How the script works

```
1. Verify avrdude.exe, avrdude.conf and the .hex file all exist   (fail early)
2. If the board is in normal mode, trigger the bootloader with a
   "1200-baud touch" (opening the port at 1200 baud resets a Leonardo)
3. Poll every 200 ms for VID_2341&PID_0036 (bootloader) or a port
   whose name contains "bootloader"
4. The moment it appears, run:
     avrdude -C <conf> -p atmega32u4 -c avr109 -P \\.\COM9 -b 57600 -D \
             -U flash:w:EMUXONE.hex:i
5. Print the result and the post-flash device state
```

avrdude flags explained:

| Flag | Meaning |
|---|---|
| `-p atmega32u4` | The chip on the Leonardo |
| `-c avr109` | The bootloader protocol (Caterina/CATERIN uses AVR109) |
| `-P \\.\COM9` | The port. `\\.\` is a Windows prefix needed for COM10+ |
| `-b 57600` | Bootloader baud rate |
| `-D` | Don't erase the whole chip — preserves the bootloader |
| `-U flash:w:file:i` | Write (`w`) the Intel-hex (`i`) file to flash |

> **`-D` matters.** Without it a full chip erase can wipe the bootloader, and
> then you can no longer flash over USB at all.

---

## 7. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `TIMED OUT: no bootloader port detected` | Not plugged into the PC; charge-only USB cable; or RESET not pressed. Press RESET *while* it says it's waiting. |
| `programmer did not respond to command: select device` | The 8-second window closed. Press RESET and rerun — this is the classic failure. |
| `Access is denied` on the port | Something else holds the port. Close GIMX and the Arduino IDE. |
| Board vanishes from COM ports after flashing | **Correct behaviour** — it's a gamepad now, not a serial port. |
| Want the plain Arduino back | Reflash a normal sketch from the Arduino IDE. The bootloader survives (thanks to `-D`), so this is always recoverable. |

---

## 8. Is it safe?

Yes, and it's reversible. Because we pass `-D`, the bootloader is preserved, so
you can always flash something else later. The worst realistic outcome is a
failed write, which you fix by pressing RESET and running the script again.

---

**Next:** [03 — Console Connector](03-console-connector-docs.md)
