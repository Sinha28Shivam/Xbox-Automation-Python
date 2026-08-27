# 08 — Roadmap: The Agentic Testing Framework

Where this project is going: an AI-driven framework that tests consoles and
launches games automatically, and — critically — **verifies what it did by
looking at the screen**.

---

## 1. The goal

Give the system an instruction in plain language:

> *"Launch Forza Horizon and confirm it reaches the main menu."*

and have it navigate the dashboard, start the game, watch the screen, and report
pass or fail **with screenshot evidence**.

---

## 2. Why we need a capture card

Everything built so far is **action without perception**. We can send button
presses, but the software has no idea what's on the TV.

That's not a small gap — it's the difference between automation and testing. As
documented in doc 07, a script that only sends commands can report six
consecutive successes while the console does nothing. **Scaling that up would
just produce false passes faster.**

An agent needs the loop:

```
   OBSERVE  ->  DECIDE  ->  ACT  ->  OBSERVE (again, to confirm)
      ^                                  |
      +----------------------------------+
```

Right now we only have ACT. The capture card supplies OBSERVE.

### Environment status

| Capability | Status |
|---|---|
| Action (button presses) | ✅ Working and verified on hardware |
| LLM reasoning | ✅ `ANTHROPIC_API_KEY` set; `anthropic`, `openai`, `langchain` installed |
| Vision / OCR | ✅ `opencv`, `paddleocr`, `pytesseract`, `pillow`, `numpy` installed |
| Video capture | ✅ `ffmpeg` 9.0 installed and DirectShow enumeration works |
| **Capture hardware** | ✅ **AVerMedia ExtremeCap UVC — VERIFIED WORKING** |
| Test harness | ✅ `pytest`, `pytest-asyncio` installed |

**Every prerequisite is now met.** The capture card was tested end to end: a
1920x1080 frame of the real Xbox dashboard was captured programmatically and
visually confirmed. Full details in [10 — Capture Card](10-capture-card-docs.md).

Key findings from that test:
* The card is standard **UVC** — no vendor SDK, and **ReCentral 4 is not needed**
  (in fact it must be closed, as it holds the device exclusively).
* **OpenCV index 1** is the card; index 0 is the laptop webcam, which returns
  plausible-looking frames and could silently fool a naive check.
* After the device is open, frames cost **~1 ms** — cheap enough to verify after
  every button press.

---

## 3. Architecture

Five layers, each testable on its own:

```
┌─────────────────────────────────────────────────────┐
│  5. REPORTING     artifacts, screenshots, JUnit XML │
├─────────────────────────────────────────────────────┤
│  4. AGENT         Claude vision decides next action │
├─────────────────────────────────────────────────────┤
│  3. VERIFICATION  did the screen actually change?   │
├─────────────────────────────────────────────────────┤
│  2. PERCEPTION    capture frames, OCR text          │
├─────────────────────────────────────────────────────┤
│  1. ACTION        ConsolePad  (BUILT & WORKING)     │
└─────────────────────────────────────────────────────┘
```

Proposed layout:

```
framework/
  action/       pad.py            <- from test_controller.py
                udp_sender.py     <- faster persistent sender
  perception/   capture.py        <- grab frames from the card
                ocr.py            <- read on-screen text
  state/        verify.py         <- did the action have an effect?
                screens.py        <- recognise known screens
  agent/        loop.py           <- ReAct loop with Claude vision
                tools.py          <- press_button, read_screen, find_text, wait
  report/       artifacts.py      <- frames, logs, timings
tests/          conftest.py, test_launch_game.py
config/         controls.yaml (exists), screens/
```

---

## 4. The layers in detail

### Layer 1 — Action ✅ built
`ConsolePad` already provides `press`, `press_times`, `hold`, `stick`, `trigger`,
`sequence`, `run_macro`, plus config-driven names and an action log.

**One improvement planned:** each press currently spawns a `gimx.exe` process
(~250 ms). Fine for menus, too slow for timing-sensitive input. A persistent UDP
sender speaking GIMX's protocol directly would fix this, with the subprocess path
kept as a fallback.

### Layer 2 — Perception ⏳ unblocked, ready to build
```python
capture.grab_frame()            # one frame as a numpy array
capture.wait_for_stable_screen() # wait until animation stops (frame differencing)
ocr.read_text(frame)            # PaddleOCR primary, pytesseract fallback
```

`wait_for_stable_screen()` is what removes guessy `sleep()` calls. Instead of
"wait 3 seconds and hope," we wait until the picture stops changing.

### Layer 3 — Verification ⏳ the anti-false-pass core

**The most important layer**, and the direct answer to this project's biggest
recurring problem:

```python
def verify_action_had_effect(pad, capture, action):
    before = capture.grab_frame()
    action()
    capture.wait_for_stable_screen()
    after = capture.grab_frame()
    if images_are_identical(before, after):
        raise AssertionError("Screen did not change - the action had no effect")
```

Also: `wait_for_text("Play")`, `detect_screen("xbox_home")` via perceptual
hashing against reference images.

> **Design rule:** "GIMX accepted the event" must **never** count as a pass. Only
> observed pixel change does.

### Layer 4 — Agent ⏳
A ReAct loop using Claude's vision. Tools exposed: `press_button`, `read_screen`,
`find_text`, `wait`. Given a goal, it looks at the frame, picks the next press,
acts, then re-observes.

Guardrails: max step count, wall-clock timeout, and a hard rule that any success
claim must cite visual evidence.

### Layer 5 — Reporting ⏳
Per-run artifacts — frames, action log, timings, pass/fail with screenshots —
plus JUnit XML so CI can consume results.

---

## 5. Build order

| Phase | Work | Blocked? |
|---|---|---|
| **1** | Refactor `ConsolePad` into a library package; pytest fixtures that verify the GIMX session before any test runs | No |
| **2** | `capture.py` + `ocr.py`; auto-detect the device | No — card verified |
| **3** | `verify.py` — the anti-false-pass core | No |
| **4** | Agent loop with Claude vision + guardrails | No |
| **5** | Reporting and CI | No |

**Nothing is blocked.** With capture verified, all five phases can proceed.
Phase 2 already has a working reference implementation in
`temp_code_test/hardware/capture_probe.py` (device selection, warmup, blank-frame detection and
frame differencing).

---

## 6. Multi-console design

Build around **console profiles** from day one. `controls.yaml` already has
entries for `xbox_one`, `ps4`, `ps3` and `xbox_360`, and GIMX ships matching
firmware (EMUPS4, EMUPS3, EMU360). Adding PlayStation later should be a config
entry plus a reflash:

```powershell
.\flash_leonardo.ps1 -Firmware EMUPS4
```

Cheap to design in now; expensive to retrofit.

> Only `xbox_one` is hardware-tested. The other profiles come from GIMX's
> documented types and are **unproven here.**

---

## 7. Realistic expectations

**Game launch verification is harder than menu navigation.**

- Load times vary by title, install state and console model
- Splash screens differ wildly
- First runs can show unexpected dialogs (updates, EULAs, profile pickers)
- Some titles need network, which introduces its own flakiness

Plan for generous timeouts and per-game reference screens, and treat the first
few games as **calibration** rather than assuming the approach generalises. Our
current `game_launch_wait: 30.0` is a placeholder, not a measurement.

**Also worth flagging:** OCR on video-game UIs is harder than on documents —
stylised fonts, animated backgrounds, transparency and motion blur all hurt
accuracy. Expect to combine OCR with image hashing rather than relying on text
alone.

---

## 8. Open questions

1. ~~Capture resolution/latency~~ — **ANSWERED.** 1920x1080, ~1 ms per frame
   once the device is open. Fast enough to verify after every button press.
2. **Reference screenshots** — the library of "known screens" has to be built by
   hand once, per dashboard version.
3. **Cost control** — every agent step is a vision API call. We'll want caching
   and cheap pre-checks (image hashing) before invoking the model.
4. **Flaky-test policy** — how many retries before a genuine failure is declared?

---

## 9. Success criteria

The framework is working when it can:

- [x] Capture frames reliably from the console  *(verified: 1080p, real content)*
- [ ] Detect that the screen changed after an action
- [ ] Read on-screen text well enough to find menu items
- [ ] Navigate from dashboard to a named game, unaided
- [ ] Confirm the game reached its main menu **from the image**
- [ ] Produce a pass/fail report with screenshot evidence
- [ ] **Fail correctly** when the console doesn't respond

That last point matters most. A framework that can't fail is worthless — which is
the lesson this whole project has been teaching.

---

**Back to:** [00 — Start Here](00-README-START-HERE.md)
