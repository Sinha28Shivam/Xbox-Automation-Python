# Xbox Console Automation

Control a console with an Arduino Leonardo emulating a controller — and, on top
of that, an **agentic testing framework** that takes a test written in plain
English, drives the console, watches the screen, and reports pass or fail with
screenshot evidence.

Two projects, two layers:

| Project | Layer | What it does |
|---|---|---|
| **[Xbox-Automation-Python](#4-the-hardware-layer)** | Hardware | Sends input, captures video. Verified on real hardware. |
| **[Xbox-Agentic-Testing](#5-the-agentic-layer)** | Orchestration | LangGraph agents that plan, execute, verify and diagnose. |

The agentic layer contains **no hardware code** — it imports the modules above
at runtime, so `controls.yaml` stays the single source of truth.

---

## 1. System Architecture

```
   PLAIN-ENGLISH SCENARIO          "press A and confirm the screen changes"
          │
          ▼
   AGENTIC FRAMEWORK               health -> plan -> execute -> verify -> report
          │        ▲
          │        └──────── capture card ◄─── HDMI ◄──┐
          ▼                                            │
   gimx.exe (CLIENT)                                   │
          │  UDP 127.0.0.1:51914                       │
          ▼                                            │
   gimx.exe (SERVER)               owns the serial port│
          │  USB                                       │
          ▼                                            │
   FTDI UART adapter               USB <-> serial (COM8)
          │  TX / RX                                   │
          ▼                                            │
   ARDUINO LEONARDO                GIMX firmware       │
          │  USB (looks like a controller)             │
          ▼                                            │
   XBOX CONSOLE ───────────────────────────────────────┘
```

The loop closing back through the capture card is the important part. Without
it the system can only *act*; with it, it can **observe whether the action
worked** — which is the difference between automation and testing.

- **Arduino Leonardo** — its ATmega32U4 has native USB, so it can present
  itself to the console as a real controller.
- **GIMX** — translates PC events into controller USB packets, and supplies
  the Leonardo firmware.
- **AVerMedia ExtremeCap UVC** — a standard UVC capture card. No vendor SDK,
  and RECentral must be **closed** (only one app can hold a capture device).

---

## 2. Directory Layout

```
xboxArudino/
├── README.md                        <- you are here
├── docs/                            <- guides, hardware notes, post-mortems
│   ├── 00-README-START-HERE.md
│   ├── 01-hardware-setup.md         <- wiring diagrams & parts list
│   ├── 02-leonardo-flash-docs.md    <- flashing the firmware
│   ├── 03-console-connector-docs.md <- connection troubleshooting
│   ├── 04-buttons-docs.md           <- friendly names -> GIMX names
│   ├── 05-test-controller-docs.md   <- CLI / library usage
│   ├── 06-troubleshooting.md        <- solutions for common errors
│   ├── 07-lessons-learned.md        <- READ THIS. Why the framework is built
│   │                                   the way it is.
│   ├── 08-roadmap-agentic-framework.md
│   ├── 09-gimx-session-docs.md
│   └── 10-capture-card-docs.md
│
├── Xbox-Automation-Python/          <- HARDWARE LAYER
│   ├── config/controls.yaml         <- single source of truth: buttons,
│   │                                   timings, capture, console profiles
│   ├── capture/capture.py           <- capture card, frame stats, blank
│   │                                   detection, device resolution by name
│   ├── console-connector/           <- port scan & connection diagnosis
│   ├── flash-leonardo/              <- firmware flashing script
│   ├── gimx-session/                <- session lifecycle, serial port, auth
│   └── test-controller/             <- ConsolePad: config-driven input
│
└── Xbox-Agentic-Testing/            <- AGENTIC LAYER
    ├── console.py                   <- the CLI
    ├── config/                      <- settings, agent roster, graph topology,
    │                                   prompts. Nothing is hardcoded.
    ├── core/                        <- config, schemas, state, LLM factory,
    │                                   hardware adapters, artifact store
    ├── tools/                       <- what agents can do: hardware probes,
    │                                   input, vision/OCR, reporting
    ├── agents/                      <- the nine agents
    ├── graph/                       <- routing, graph builder, runner
    ├── scenarios/                   <- example tests
    └── artifacts/                   <- per-run frames, logs, reports (ignored)
```

---

## 3. Quick Start

```bash
pip install -r Xbox-Automation-Python/requirements.txt
pip install -r Xbox-Agentic-Testing/requirements.txt
```

**Terminal 1** — start the GIMX server (it holds the serial port):

```bash
python Xbox-Automation-Python/gimx-session/gimx_session.py start
```

> [!IMPORTANT]
> **Hold the controller's GUIDE button for 2 seconds** when prompted.
> Without it GIMX accepts every event, reports `ok`, and delivers nothing to
> the console. This is the single most common cause of a test that appears to
> work while the console sits idle.

**Terminal 2** — check the rig, then run something:

```bash
# Hardware layer: is a session reachable?
python Xbox-Automation-Python/test-controller/test_controller.py --check

# Agentic layer: full rig check (no API key needed)
python Xbox-Agentic-Testing/console.py health

# Run a test written in plain English
python Xbox-Agentic-Testing/console.py run "Press A and confirm the screen changes"
```

---

## 4. The Hardware Layer

Direct control, no AI. Useful on its own and for debugging the rig.

```bash
cd Xbox-Automation-Python

# every mapped button, trigger, stick and macro
python test-controller/test_controller.py --list

# send presses (repeats supported)
python test-controller/test_controller.py press down*3 a

# run a macro from controls.yaml
python test-controller/test_controller.py macro nav_test

# capture card: is it working and is there a signal?
python capture/capture.py preflight
python capture/capture.py grab --out shot.png
```

> [!NOTE]
> A successful send only proves **GIMX accepted the event**. It does not prove
> the console reacted. That gap is exactly what the agentic layer exists to
> close.

---

## 5. The Agentic Layer

Nine agents, wired together with LangGraph:

```
health ──healthy──> scenario_validator ──> planner ──> executor
  │                                           ↑            │
  └──blocked──> reporter                      │       (evidence)
                   ↑                          │            ↓
                   │                       replan       verifier
                   └───────── passed ─────────┴──failed──> rca
```

| Agent | Job |
|---|---|
| health | Is the rig usable? GIMX, capture card, adapters |
| scenario_validator | English or YAML → a testable scenario |
| planner | Scenario → steps, each with an expected observation |
| executor | Drive the console, capture before/after frames |
| verifier | Judge the evidence. Adversarial by design |
| rca | Rig fault or product defect? |
| reporter | JSON / Markdown / JUnit with screenshots |
| supervisor | Dynamic routing (optional) |
| recovery | Bounded remediation (optional) |

```bash
cd Xbox-Agentic-Testing

python console.py health                       # check the rig
python console.py info                         # adapters, provider, agents
python console.py run "Open the guide"         # plain English
python console.py run --file scenarios/dashboard-navigation.yaml
python console.py run "..." --dry-run          # plan without touching hardware
python console.py interactive                  # REPL
```

Exit codes: `0` pass · `1` fail · `2` **blocked (rig broken)** · `3`
inconclusive · `4` framework error.

Full details, configuration and troubleshooting:
**[Xbox-Agentic-Testing/README.md](Xbox-Agentic-Testing/README.md)**

### The rule it is built around

From [docs/07-lessons-learned.md](docs/07-lessons-learned.md): the early
scripts once reported **six consecutive successes while the console did
nothing.** So:

> **"The command was accepted" is never evidence that anything happened.**
> Only observed pixels count.

That is enforced in `core/schemas.py` — a `PASS` with no observational
evidence is automatically downgraded to `INCONCLUSIVE` by a pydantic
validator, not by a prompt a model can ignore.

---

## 6. Safety Warning

> [!WARNING]
> **Do not run GIMX as administrator unless absolutely necessary.**
> Elevated, `gimx.exe` claims REALTIME CPU priority and grabs your input
> devices. During development this froze an entire PC — stuck mouse, nothing
> clickable.
>
> `gimx-session/gimx_session.py` prevents a repeat: it always passes
> `--nograb`, sets an idle `--timeout`, and pins the process priority back to
> Normal immediately after launch. Those protections apply only to sessions
> **it** starts.

---

## 7. Verification

Everything below runs offline — no console, no capture card, no API key:

```bash
# documentation links resolve
python docs/_verify_docs.py

# hardware layer configuration
python Xbox-Automation-Python/gimx-session/_selftest.py
python Xbox-Automation-Python/test-controller/_selftest.py

# agentic layer: wiring, schemas, graph, routing
python Xbox-Agentic-Testing/_smoke_test.py

# regression tests for bugs found against real hardware
python Xbox-Agentic-Testing/_test_fixes.py
```

With the rig connected:

```bash
python Xbox-Agentic-Testing/console.py health        # full rig check
python Xbox-Agentic-Testing/_diagnose_capture.py     # probe every video device
python Xbox-Agentic-Testing/_check_ocr.py            # is OCR usable?
python Xbox-Agentic-Testing/_inspect_run.py          # summarise the last run
python Xbox-Agentic-Testing/_verify_report.py        # do report images resolve?
```

---

## 8. Known Limitations

Worth reading before trusting a result:

- **Reachable ≠ authenticated.** GIMX answers UDP whether or not anyone held
  the Guide button. Only an observed screen change settles it.
- **OpenCV device indices shift.** The capture card moved from index 1 to 0
  when USB devices changed, and the framework silently read the laptop webcam
  instead. Devices are now resolved **by name**; run `_diagnose_capture.py` if
  frames look black.
- **OCR on game UIs is unreliable.** Stylised fonts over animated backgrounds.
  A missed string is weak evidence of absence, and the tools say so.
- **HDCP content captures black.** Streaming apps cannot be verified at all.
- **`game_launch_wait: 30.0` is a placeholder, not a measurement.** Real launch
  times vary enormously by title and install state.
- **Only `xbox_one` is hardware-verified.** The PS4/PS3/360 profiles come from
  GIMX's documented types and are unproven here.
- **One run at a time.** The capture card can only be held by one process.
