# 01 — Hardware Setup

What the physical parts are, why each one is needed, and how they connect.

---

## 1. Parts list

| Part | Purpose | What we used |
|---|---|---|
| **Arduino Leonardo** | Pretends to be an Xbox controller | ATmega32U4 board |
| **FTDI USB-UART adapter** | Lets the PC talk to the Leonardo over serial | FTDI FT232 (`VID_0403 / PID_6001`) |
| **Xbox One controller** | Authenticates the session (see below) | `VID_045E / PID_02D1` |
| **PC** | Runs GIMX and our Python scripts | Windows 11 |
| **Xbox console** | The device under test | — |
| **HDMI capture card** | *Not yet installed* — needed for screen verification | see doc 08 |

---

## 2. Why each part exists

### Arduino Leonardo — the fake controller
A console will not take input from a random USB device. It only trusts a real
controller. The Leonardo's **ATmega32U4** chip has USB hardware built in, so it
can *impersonate* a USB gamepad. An Arduino Uno **cannot** do this — its USB is
handled by a separate chip you can't easily repurpose. This is why the board
choice matters.

Once flashed with GIMX firmware, the console sees the Leonardo as a controller
and accepts its button presses.

### FTDI UART adapter — the PC's voice
The Leonardo's USB port is busy pretending to be a controller *for the console*.
It cannot simultaneously be a normal Arduino serial port for the PC. So we need
a **second, separate channel** for the PC to send instructions: that's the FTDI
adapter, which converts USB (PC side) into plain serial TX/RX wires (Leonardo
side).

This is the part people find confusing, so to be explicit: **the Leonardo has
two connections** — USB to the console, and serial wires to the FTDI adapter.

### Xbox controller — the authenticator
An Xbox One console uses a security handshake. GIMX handles this by
"pass-through": a genuine Microsoft controller stays plugged into the PC and
supplies the authentication. In the GIMX log you'll see:

```
found pass-through device 0x045e:0x02d1
```

**This is why you must hold the GUIDE button for 2 seconds when starting a
session.** It is not optional and not a quirk of our scripts — without it, GIMX
runs and accepts events but nothing reaches the console.

---

## 3. Wiring

```
   PC  ──USB──>  FTDI adapter  ──TX/RX/GND──>  Arduino Leonardo  ──USB──>  Xbox
   PC  ──USB──>  Xbox One controller (authentication only)
```

Serial wiring between FTDI and Leonardo — note the crossover:

| FTDI pin | Leonardo pin | Why |
|---|---|---|
| TX  | RX (D0) | One device's *transmit* goes to the other's *receive* |
| RX  | TX (D1) | And vice versa |
| GND | GND | **Required.** Both sides need a shared voltage reference |

> **Common mistake:** connecting TX→TX and RX→RX. Nothing will work and there is
> no error message — just silence. If in doubt, swap the two data wires.
>
> **Also easy to miss:** forgetting GND. Serial communication *cannot* work
> without a common ground, even though both devices have their own USB power.

---

## 4. Verifying what's connected

List every COM port currently present:

```powershell
powershell -Command "[System.IO.Ports.SerialPort]::GetPortNames()"
```

Expected on this setup: **COM3** (Intel AMT, irrelevant) and **COM8** (our FTDI
adapter).

For more detail, including USB IDs:

```powershell
powershell -Command "Get-PnpDevice -PresentOnly -Class Ports | Select-Object Status,FriendlyName,InstanceId | Format-List"
```

> ### Important: ghost devices
> Windows Device Manager remembers devices that were *ever* plugged in. When we
> first listed ports we saw an Arduino Due on COM6/COM7 and a CH340 on COM5 —
> **none of which were actually connected.** Always use the `-PresentOnly`
> flag, or you will waste time chasing hardware that isn't there.

Reference values for this build:

| Device | Identifier | Port |
|---|---|---|
| FTDI UART | `VID_0403&PID_6001`, serial `A50285BI` | COM8 |
| Leonardo (bootloader mode only) | `VID_2341&PID_0036` | COM9 |
| Xbox One controller | `VID_045E&PID_02D1` | — |

Note the Leonardo only appears as a COM port while in **bootloader mode**
(during flashing). Once running GIMX firmware it presents itself as a gamepad,
not a serial port — so *its absence from the port list is normal and correct*.

---

## 5. Power and boot order

The Leonardo draws power from whichever USB port is connected. In practice:

1. Connect the FTDI adapter to the PC.
2. Connect the Leonardo to the console (this powers it and starts the pretend
   controller).
3. Plug the real Xbox controller into the PC for authentication.
4. Start GIMX and hold GUIDE for 2 seconds.

If you flash the Leonardo, it must be connected to the **PC** at that moment,
not the console — see doc 02.

---

## 6. Software prerequisites

| Software | Version used | Notes |
|---|---|---|
| GIMX | 8.0 | Installed at `C:\Program Files\GIMX\` |
| Python | 3.11 | Any 3.9+ should work |
| PyYAML | 6.0.2 | `pip install pyyaml` |
| avrdude | 6.3 | Bundled with GIMX — no separate install |

GIMX ships everything needed for flashing (`avrdude.exe`, `avrdude.conf`, and
the firmware `.hex` files in `C:\Program Files\GIMX\firmware\`).

---

**Next:** [02 — Flashing the Leonardo](02-leonardo-flash-docs.md)
