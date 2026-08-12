# 10 — Capture Card (AVerMedia) & ReCentral 4

**Probe script:** `Xbox-Automation-Python/capture/_probe_capture.py`

**Status: VERIFIED WORKING.** A 1920x1080 frame of the real Xbox screen was
captured programmatically and visually confirmed. This unblocks the whole
perception layer.

---

## 1. The short answer about ReCentral 4

> **You do NOT need ReCentral 4 — and it must be CLOSED while automating.**

The card is a **standard UVC device**, so ffmpeg and OpenCV talk to it directly
with no vendor SDK. ReCentral is AVerMedia's consumer recording GUI; it's fine
for eyeballing the feed, but it **holds the device exclusively**.

With ReCentral running, our first capture attempt failed:

```
Could not run graph (sometimes caused by a device already in use by other application)
Error opening input: I/O error
```

Closing the ReCentral window fixed it immediately. Only **one** application can
open a capture device at a time.

**So ReCentral's role is: a handy viewer for setup, and nothing more.** Use it to
confirm the console is outputting video, then close it before running anything.

---

## 2. What the hardware reports

| Property | Value |
|---|---|
| Name (ffmpeg/dshow) | `AVerMedia ExtremeCap UVC` |
| USB ID | `VID_07CA&PID_0110` |
| Device class | Camera (UVC) — no special driver needed |
| Max resolution | **1920x1080** |
| Frame rates | up to **60 fps** (25/30/60 depending on mode) |
| Codec | MJPEG |
| Audio | separate device: `AVerMedia ExtremeCap UAC` |

It exposes many modes (640x480 up to 1920x1080). We use 1080p30 — 60 fps buys
nothing for menu automation and doubles the data.

---

## 3. Verified results

```
--- ffmpeg backend ---
  grab 0 (0.66s): 1920x1080  mean= 64.49  std= 49.46  tones=256  -> has content
  grab 1 (0.81s): 1920x1080  mean= 64.49  std= 49.46  tones=256  -> has content

--- OpenCV backend (index 1) ---
  reports 1920x1080
  frame 2: 1920x1080  mean= 64.36  std= 49.30  tones=255  -> has content
  grab latency: min 0.6 ms  avg 146.0 ms  max 726.0 ms
```

Both backends agree (std ≈49.3–49.5, 255–256 tones), and the user confirmed the
saved PNG is the Xbox dashboard.

---

## 4. ⚠️ The index trap — read this

**OpenCV index 0 is the laptop webcam. Index 1 is the capture card.**

This is genuinely dangerous, because index 0 **does not fail**. It returns
perfectly valid 1280x720 frames of a dark room, which a naive check would
happily call "has content". You could build an entire verification layer on top
of your own webcam and not notice for hours.

How to tell them apart:

| | Index 0 (webcam) | Index 1 (capture card) |
|---|---|---|
| Resolution | 1280x720 | **1920x1080** |
| Std deviation | ~1.6 (dark, flat) | **~49 (real content)** |
| Unique tones | ~16 | **255** |

**Mitigations we use:**
1. `opencv_index: 1` is recorded in `controls.yaml` with a comment saying why.
2. The ffmpeg path matches by **device name**, which can't drift the way indices
   can.
3. The probe reports resolution and std so a wrong device is obvious.

If you add or remove any USB camera, **re-run the probe** — indices can shift.

---

## 5. Which backend to use

| | OpenCV | ffmpeg |
|---|---|---|
| First frame | ~726 ms (opening the device) | ~0.7 s |
| Subsequent frames | **~1 ms** | ~0.7 s (new process each time) |
| Selection | by index (fragile) | by **name** (stable) |
| Best for | the agent loop | one-off grabs, diagnostics |

**Recommendation:** keep **one OpenCV handle open** for the agent loop — after
the initial open, frames are essentially free (~1 ms), which makes
capture→act→capture verification cheap. Use ffmpeg for standalone screenshots
and when you want name-based selection.

Note the first frames after opening can be blank while the card syncs, hence
`warmup_frames: 3` in the config.

---

## 6. Running the probe

```bash
python Xbox-Automation-Python/capture/_probe_capture.py
python Xbox-Automation-Python/capture/_probe_capture.py --index 1
python Xbox-Automation-Python/capture/_probe_capture.py --backend ffmpeg
```

It saves frames to `capture/_probe_frames/` so you can look at them.

**Why the probe checks pixels, not files.** A black frame is still a valid file.
Writing 1.4 MB proves nothing about whether there's a picture in it. So the probe
computes mean, standard deviation and unique tone count, and compares consecutive
frames to see whether the feed is actually *live*.

This is the same principle that burned us with the serial port: a successful
write is not evidence of an effect. Checking the output is.

---

## 7. Configuration

Recorded in `config/controls.yaml` under `capture:`

```yaml
capture:
  device_name: "AVerMedia ExtremeCap UVC"
  usb_id: "VID_07CA&PID_0110"
  opencv_index: 1          # VERIFIED. 0 = built-in webcam, not the card
  width: 1920
  height: 1080
  fps: 30
  backend: "opencv"
  warmup_frames: 3
  blank_std_threshold: 1.0
  conflicting_apps: ["RECentral 4", "OBS", "Teams", "Camera", "Zoom"]
```

`blank_std_threshold: 1.0` sits between a measured blank frame (std **0.0**) and
measured real content (std **~49**) — a very wide margin.

---

## 8. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `Could not run graph … device already in use` | **Close ReCentral 4** (also OBS, Teams, Camera, Zoom) |
| Frames are `BLANK/FLAT` (std 0) | No signal. Console off, HDMI in the wrong port, or HDCP |
| Wrong picture / dark room | You're on the **webcam** (index 0). Use index 1 |
| `No JPEG data found in image` | Harmless MJPEG warning during startup; the frame still decodes |
| Resolution is 1280x720 not 1080 | Wrong device index, or the console outputs 720p |
| Works once then fails | Another app grabbed the device in between |

### About HDCP
Xbox encrypts protected content (Netflix, some media apps), which appears as a
**black frame** on any capture device. The dashboard and most games are normally
fine. If a specific app captures black while everything else works, HDCP is the
likely cause — not a bug in our code.

---

## 9. What this unblocks

With verified capture, the remaining layers from the roadmap become buildable:

- **Perception** — `grab_frame()`, `wait_for_stable_screen()`, OCR
- **Verification** — `verify_action_had_effect()`: capture → act → capture → diff
- **Agent** — Claude vision decides the next button from the actual screen
- **Reporting** — pass/fail with screenshot evidence

The measured ~1 ms per frame (after open) means the observe→act→observe loop is
fast enough to run after every single button press.

---

## 10. Setup checklist

1. HDMI from the console into the card's **IN** port
2. HDMI from the card's **OUT** to your TV (pass-through)
3. Card's USB into the PC
4. Console powered on, showing the dashboard
5. **Close ReCentral 4**
6. Run the probe and confirm real content at 1920x1080

---

**Related:** [08 — Roadmap](08-roadmap-agentic-framework.md) ·
[01 — Hardware Setup](01-hardware-setup.md) ·
[00 — Start Here](00-README-START-HERE.md)
