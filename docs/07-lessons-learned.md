# 07 — Lessons Learned

An honest record of what went wrong during development, why, and what we changed.
Most of these mistakes are easy to repeat, which is exactly why they're written
down.

---

## Lesson 1: A successful `write()` proves nothing

### What we did
The first implementation opened COM8 with pyserial and wrote raw bytes:

```python
ser = serial.Serial("COM8", 500000)
ser.write(packet)     # <-- always "succeeds"
print("  -> down   ok")
```

Output looked perfect:

```
Connected on COM8 @ 500000
  -> down (0.08s)
  -> right (0.08s)
Done.
```

**Nothing happened on the console. Not one press registered.**

### Why it fooled us
Opening a serial port and writing to it succeeds **whether or not anything is
listening on the other end**. The OS hands your bytes to the UART chip and
returns success. If they disappear into nothing, nobody reports an error.

So the code was "working" — locally — while achieving nothing at all. And it was
reported as working, which was worse than the bug itself.

### The fix
Write a probe that **reads a reply**:

```python
ser.write(query)
data = ser.read(64)      # <-- the line that turns a guess into evidence
```

Sweeping seven baud rates gave silence at every one. That single result proved
the whole approach was impossible and redirected the work.

### Takeaway
> **Never report success from the sending side alone.** A test that cannot fail
> is not a test. If you can't observe the effect, say clearly that you haven't
> verified it.

---

## Lesson 2: Read the tool's own error messages literally

GIMX printed:

```
Highest priority class can't be used due to missing elevation.
```

This was read as *"needs administrator to work"* and the advice given was "run as
administrator." **That froze the entire PC** — stuck mouse, nothing clickable.

The message actually meant: elevation would let GIMX claim **realtime CPU
priority**. Given it, GIMX took that priority *and* grabbed input devices,
starving Windows' own input threads.

### Takeaway
> A message about a *capability* is not a request for permission. Before
> recommending privilege escalation for a realtime input tool, consider what it
> will do with that privilege. This mistake caused real disruption to the user's
> machine.

Mitigations now in place: `--nograb`, `--timeout 5`, forced Normal priority, and
an explicit "do not run as administrator" warning in the code and docs.

---

## Lesson 3: Don't guess API names — verify them

The button map was written from what looked reasonable:

```python
"guide": "guide",   # WRONG - GIMX rejects "guide"
"lb": "LB",         # WRONG - must be "l1"
```

Some of these names even *appear* inside GIMX's own DLL, which made them look
right. But the event parser **rejects** them.

Testing every candidate against `gimx.exe` produced the truth:

```
REJECTED : guide, LB, RB, LT, RT, menu, view, back, gas, brake, lstick right
accepted : PS, l1, r1, l2, r2, start, select, lstick x, lstick y
```

The Guide button would have silently failed forever.

### Takeaway
> Strings existing in a binary doesn't mean they're valid inputs. Verify against
> the actual interface. A cheap validation loop beats plausible-looking
> assumptions.

Handy trick — validate a name without sending anything, by aiming at a dead port:

```powershell
& 'C:\Program Files\GIMX\gimx.exe' --type XOnePad --event 'NAME(1)' --dst 127.0.0.1:59999
```

---

## Lesson 4: Read the tool's documentation before reverse-engineering it

Considerable effort went into hand-building 18-byte Xbox input reports and
guessing bit layouts. Meanwhile `gimx.exe --help` contained:

```
--event "control(value)": send controls to the console and exit.
```

A documented, supported interface for exactly the task. Its help text even
warned:

```
The --event argument may require running two gimx instances.
```

That sentence was the answer to the architecture question.

### Takeaway
> Check `--help` and the docs **first**. Reverse-engineering a protocol should be
> the last resort, not the first instinct.

---

## Lesson 5: One missing argument can look like a hardware failure

A GIMX session was started without `--config`:

```powershell
gimx.exe --type XOnePad --src 127.0.0.1:51914 --port COM8    # no --config
```

It reported `GIMX adapter detected`, `Firmware version: 8.0`, negotiated
2000000 baud — everything looked healthy. But **no input reached the console at
all.** Without a config file there are no input bindings.

### How we found it
By asking the user to try the **physical controller**. It didn't work either.
That immediately proved the fault was in the *session*, not the script.

### Takeaway
> When automation fails, test whether the manual path works under identical
> conditions. It splits the problem space in half instantly. This one question
> was the most efficient debugging step in the whole project.

---

## Lesson 6: Argument order can matter

```
controller #1: option -t with value `XOnePad'
now reading arguments for controller #2      <-- everything after went here
```

GIMX's docs state: *"A --bdaddr, --port or --dst argument finishes the current
controller options."* So `--port` **must come last**, or the remaining options get
assigned to a second, non-existent controller.

### Takeaway
> Some CLIs are positional in non-obvious ways. When arguments seem ignored, read
> the parser's own diagnostic output — GIMX was clearly reporting the problem.

---

## Lesson 7: Filter to *present* devices

The first port scan showed an Arduino Due on COM6/COM7 and a CH340 on COM5 — none
of them plugged in. Windows remembers every device ever attached.

```powershell
Get-PnpDevice -PresentOnly ...     # <-- the flag that matters
```

### Takeaway
> Distinguish "known to the system" from "currently connected" before reasoning
> about hardware.

---

## Lesson 8: Respect timing windows in hardware

The Leonardo bootloader runs for **~8 seconds** after reset. A manual flash
attempt failed mid-operation:

```
Found programmer: Id = "CATERIN"
error: programmer did not respond to command: select device
```

Not a configuration problem — the window closed. The fix was to **poll every
200 ms and fire the instant the port appears**, removing the human race
condition.

### Takeaway
> When hardware has a timing window, automate the reaction. Don't ask a person to
> beat a stopwatch.

---

## Lesson 9: Small ergonomic gaps matter

Repeats had to be typed out:

```bash
press down down down    # tedious, and easy to miscount
```

The user asked how to go "2x or 3x down." Adding `press_times()` plus flexible
syntax (`down*3`, `down 3`, `--times`, YAML `times:`) made real navigation
practical.

`interval` was added at the same time for a specific reason: menus animate, and
presses sent too fast **get swallowed**, so 3 requested downs may move the cursor
only once.

### Takeaway
> Watch for friction in how the tool is actually used. And when adding a feature
> to drive real hardware, think about what the hardware does between inputs.

---

## Lesson 10: Mark unverified things as unverified

Some sequences in `controls.yaml` are educated guesses:

```yaml
screenshot:
  verified: false     # never confirmed on hardware
guide_auth:
  verified: true      # confirmed working
```

The screenshot, clip-record and power-menu sequences follow Xbox conventions but
were never tested. Rather than presenting them as working, they carry a flag and
the code prints a warning.

Same for timing values: `game_launch_wait: 30.0` is a placeholder, not a
measurement.

### Takeaway
> Label the difference between *verified* and *assumed* in the artifact itself.
> Someone inheriting this project shouldn't have to guess which is which.

---

## What actually worked well

**Empirical validation loops.** Testing every control name against the real tool
found bugs that review would have missed.

**Asking the user to observe the TV.** The only ground truth available. Their "no,
still nothing" answers were more valuable than any log line.

**Testing the manual path.** Asking whether the physical controller still worked
was the single most efficient diagnostic step.

**Keeping the failed probe.** `console_connector.py` earned its place by
*disproving* an approach. Negative results are results.

**Dry-run mode.** Being able to inspect exactly what would be sent, with no
hardware, made iteration fast and safe.

---

## The one rule to carry into the next phase

> **Distinguish "the command was accepted" from "the desired effect happened."**

Almost every problem here traces back to that gap. It's also precisely why the
agentic framework (doc 08) is built around a capture card and screen
verification: so success can be *observed* rather than *assumed*.

---

**Next:** [08 — Roadmap: Agentic Framework](08-roadmap-agentic-framework.md)
