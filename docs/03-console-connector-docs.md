# 03 — Console Connector (Connection Diagnostics)

**Script:** `Xbox-Automation-Python/console-connector/console_connector.py`

A diagnostic tool that answers one question: **is anything actually listening on
the serial port?** It is small, but the reasoning behind it is the most valuable
lesson in this project.

---

## 1. Why this tool exists

Our very first attempt at sending button presses wrote raw bytes straight to
COM8 with pyserial. It printed:

```
Connected on COM8 @ 500000
  -> down (0.08s)
  -> right (0.08s)
  -> a (0.08s)
Done.
```

Looks perfect. **Nothing happened on the console.** Not one button registered.

Here's the trap: opening a serial port and calling `write()` **succeeds locally
whether or not anything is on the other end.** The OS accepts your bytes and
hands them to the UART chip. Nobody reports an error if they vanish into
nothing. So the script "worked" while achieving absolutely nothing.

**The lesson:** *writing* proves nothing. Only a **reply** proves a link exists.
This script was written to get that reply.

---

## 2. What it does

Sends GIMX protocol query bytes and **waits for an answer**, sweeping candidate
baud rates:

```
500000, 2000000, 1000000, 250000, 115200, 57600, 9600
```

It also checks for a **TX↔RX loopback** — if your wires are bridged, you'd hear
your own bytes echoed back and could mistake that for a working device.

---

## 3. Usage

```bash
python Xbox-Automation-Python/console-connector/console_connector.py
python Xbox-Automation-Python/console-connector/console_connector.py --port COM8
```

---

## 4. Interpreting the output

### Silence everywhere

```
--- Baud rate sweep ---
    500000 : silence
   2000000 : silence
   1000000 : silence
    250000 : silence
    115200 : silence
     57600 : silence
      9600 : silence

=========== RESULT ===========
No reply at ANY baud rate.
```

This is the **actual result we got**, and it was the decisive clue. It proved
raw serial writing could never work, because the EMUXONE firmware **only
responds to GIMX's own handshake** — it ignores strangers.

That single result saved us from endlessly tuning baud rates and button bit
layouts in a direction that was fundamentally doomed.

### A reply

```
    500000 : REPLY RECEIVED
             TYPE: 01 02 03 ...
```

Something is listening and speaking the protocol.

### Loopback detected

```
  TX is wired straight to RX (loopback) - the Leonardo is
  NOT actually receiving. Check your TX/RX wiring.
```

Your two data wires are shorted together. You're talking to yourself.

---

## 5. Important: silence is normal here

**Do not panic when this tool reports silence on a correctly working setup.**

Because the firmware only answers GIMX, silence is the *expected* result even
when everything is wired perfectly. This tool tells you about the **physical
link and firmware behaviour**, not whether your automation works.

To check whether automation is working, use the other tool:

```bash
python Xbox-Automation-Python/test-controller/test_controller.py --check
```

That looks for `Remote GIMX detected`, which is the meaningful signal for
day-to-day use.

---

## 6. When to reach for this script

- The whole chain is dead and you don't know which link broke
- You suspect wiring (TX/RX swapped, missing GND)
- You've just rewired or replaced the FTDI adapter
- You want to prove the port physically exists and can be opened

If GIMX itself reports `GIMX adapter detected` and a firmware version, the
hardware link is fine and this tool won't tell you anything new.

---

## 7. What the code does

```python
BAUDS = [500000, 2000000, 1000000, 250000, 115200, 57600, 9600]

def probe(port, baud):
    ser = serial.Serial(port=port, baudrate=baud, timeout=0.6)
    ser.write(bytes([0x00, 0x00]))   # GIMX "what are you?" query
    ser.flush()
    time.sleep(0.25)
    data = ser.read(64)              # <-- the part that matters: LISTEN
    return data                      # empty = silence
```

The essential idea in one line: **`ser.read()` is what makes this a diagnostic
instead of a guess.** Our failed first attempt only ever called `write()`.

---

## 8. Future role

When the framework grows (doc 08), this file becomes the natural home for
session management:

- Locate the GIMX install and the adapter's COM port automatically
- Start a GIMX server session with the safety flags
- Confirm UDP 51914 is listening
- Report whether the session was authenticated

For now it stays a focused diagnostic, and it's worth keeping precisely because
it once disproved a plausible-looking approach.

---

**Next:** [04 — Button Configuration](04-buttons-docs.md)
