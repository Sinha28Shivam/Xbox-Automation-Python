# Xbox-Agentic-Testing

A multi-agent console testing framework. You describe a test in plain English;
a team of agents checks the rig, plans the test, drives the console, watches the
screen, judges the result, and — when something fails — works out why.

Built on **LangGraph** and **LangChain**. Nothing is hardcoded: agents, tools,
prompts, the workflow graph, the LLM provider and every threshold live in YAML.

```bash
python console.py run "Press A on the dashboard and confirm the screen changes"
```

---

## The problem this exists to solve

From `docs/07-lessons-learned.md`: the earlier scripts once reported **six
consecutive successful runs while the console did absolutely nothing.**

That happens because GIMX accepts every event whether or not the session was
authenticated (a human has to hold the Guide button for 2 seconds). Events are
accepted, they report `ok`, and they reach nothing.

So the central rule of this framework is:

> **"The command was accepted" is never evidence that anything happened.**
> Only observed pixels count.

This is not a guideline in a prompt — a model can ignore those. It is enforced
in `core/schemas.py`: a `PASS` with no observational evidence attached is
automatically downgraded to `INCONCLUSIVE` by a pydantic validator. There is no
path through the code that produces a green result on command acknowledgements
alone.

`_smoke_test.py` test 7 asserts exactly this. If it ever fails, the framework is
unsafe to trust and nothing else matters until it is fixed.

---

## The agents

| Agent | Job | LLM? |
|---|---|---|
| **health** | Is the rig usable at all? GIMX, capture card, adapters | No |
| **scenario_validator** | Turn English or YAML into a testable scenario | Yes |
| **planner** | Scenario → ordered steps, each with an expected observation | Yes |
| **executor** | Drive the console, capture before/after frames | No |
| **verifier** | Judge the evidence. Adversarial by design | Yes |
| **rca** | Why did it fail? Rig fault or product defect? | Yes |
| **reporter** | JSON / Markdown / JUnit with screenshot evidence | No |
| **supervisor** | Dynamic routing (optional, off by default) | Yes |
| **recovery** | Bounded remediation (optional, off by default) | No |

Four agents are deliberately **deterministic**. "Is the capture device present?"
has a factual answer; running a model over it would add latency, cost and a
chance of hallucination to the component whose whole job is trustworthiness.
It also means `python console.py health` works with no API key at all.

### Two separations that matter

**The executor cannot judge.** Its schema has no verdict field. An agent that
grades its own work eventually decides it did well.

**The verifier cannot press buttons.** The tool registry grants it `tag:vision`
and `tag:analysis` only. If it could send input, it could nudge the console and
then declare success. Enforced by construction, not instruction — and asserted
in smoke test 3.

---

## The workflow

```
health ──healthy──> scenario_validator ──> planner ──> executor
  │                                           ↑            │
  └──blocked──> reporter                      │       (evidence)
                   ↑                          │            ↓
                   │                       replan       verifier
                   └───────── passed ─────────┴──failed──> rca
```

Declared entirely in `config/graph.yaml`. There is no hand-written graph in the
codebase — re-wiring the workflow is a YAML edit. Disabled agents are skipped
and their edges healed automatically.

---

## Verdicts

| Verdict | Exit | Meaning |
|---|---|---|
| `pass` | 0 | Observed evidence supported every criterion |
| `fail` | 1 | We tested the console and it misbehaved |
| `blocked` | 2 | **The rig was broken — we never tested the console** |
| `inconclusive` | 3 | The evidence did not settle it |
| `error` | 4 | The framework itself broke |

`BLOCKED` is first-class and has its own exit code. A dead capture card is not
a product defect, and a CI job that cannot tell the two apart will end up
ignoring both. In JUnit output, blocked runs are `<skipped>`, never `<failure>`.

---

## Setup

```bash
pip install -r requirements.txt

# Any one provider is enough - the factory imports by name from settings.yaml
pip install langchain-anthropic     # or langchain-openai, -google-genai, -ollama
```

Set your key in the environment or a `.env` file beside `console.py`:

```
ANTHROPIC_API_KEY=sk-ant-...
```

Verify the wiring — no hardware or API key needed:

```bash
python ../temp_code_test/agentic/smoke_test.py
```

---

## Usage

```bash
python console.py health          # is the rig usable? (no API key needed)
python console.py info            # adapters, provider, agents
python console.py tools           # every tool, and which touch hardware

python console.py run "Open the guide and confirm the overlay appears"
python console.py run --file scenarios/dashboard-navigation.yaml
python console.py run "..." --dry-run      # plan without touching hardware
python console.py interactive              # REPL
```

### Before a real run

1. Start a GIMX session in its own terminal and **hold Guide for 2 seconds**:
   ```bash
   python ../Xbox-Automation-Python/gimx-session/gimx_session.py start
   ```
2. **Close RECentral 4.** Only one application can hold a capture device.
3. Wake the console.

`python console.py health` checks all three and tells you which is wrong.

---

## Artifacts

Every run writes to `artifacts/runs/<run-id>/`:

```
frames/    step-000_before-press-a.png, step-000_after-press-a.png, ...
logs/      agent errors, transcript, raw GIMX events
reports/   report.json, report.md, junit.xml
```

Frames are saved **before** the verifier sees them, so you can check the
machine's reasoning against the same pictures it used. A verdict you cannot
audit is a verdict asking to be believed on faith.

---

## Configuration

| File | Controls |
|---|---|
| `config/settings.yaml` | LLM provider, paths, thresholds, runtime limits |
| `config/agents.yaml` | The agent roster: impl, prompt, tools, enabled |
| `config/graph.yaml` | Workflow topology, edges, routing |
| `config/prompts/*.j2` | Agent behaviour |
| `../Xbox-Automation-Python/config/controls.yaml` | Buttons, timings, devices |

Values support `${ENV_VAR:default}`, expanded at load time.

### Common changes

```bash
AGENTIC_LLM_PROVIDER=openai python console.py run "..."   # switch provider
DRY_RUN=true python console.py run "..."                  # rehearse a plan
ROUTE_MODE=supervised python console.py run "..."         # dynamic routing
```

Enable the recovery agent: set `enabled: true` under `recovery` in
`agents.yaml`. Nothing else needs editing — the graph heals its own edges.

---

## Relationship to Xbox-Automation-Python

This project contains **no hardware code**. It imports the existing modules at
runtime, by path, from the location in `settings.yaml`:

| Module | Provides |
|---|---|
| `gimx-session/gimx_session.py` | Session lifecycle, serial port, auth |
| `test-controller/test_controller.py` | `ConsolePad`, config-driven input |
| `capture/capture.py` | Capture card, frame stats, blank detection |

Those modules encode hardware measurements that took real time to establish —
the card's ~1s HDMI lock, which GIMX control names are actually accepted, that
a flat frame means no signal rather than a dark scene. Copying them here would
fork that knowledge and guarantee drift. Adapters fail soft: a missing module
becomes a `BLOCKED` verdict with a readable explanation, not a stack trace.

---

## Multi-console

Nothing here is Xbox-specific. Console profiles live in `controls.yaml`
(`xbox_one`, `ps4`, `ps3`, `xbox_360`), and agents discover the control surface
at runtime rather than being told about buttons. Set `console:` in a scenario
and reflash the adapter to test a different platform.

> Only `xbox_one` is hardware-verified. The other profiles come from GIMX's
> documented types and are unproven here.

---

## Honest limitations

- **OCR on game UIs is unreliable.** Stylised fonts over animated, transparent
  backgrounds. A missed string is weak evidence of absence, and the tools say
  so in their own results.
- **A frame delta proves *something* changed, not that the *right* thing did.**
  Vision judgement is stronger but costs an API call per check.
- **`game_launch_wait: 30.0` in controls.yaml is a placeholder, not a
  measurement.** Real launch times vary enormously by title and install state.
- **HDCP-protected content captures black.** Streaming apps cannot be verified
  at all; the scenario validator rejects such tests rather than failing them.
- **Authentication cannot be automated.** Holding Guide for 2 seconds needs a
  thumb. The recovery agent stops GIMX but never silently restarts it — an
  unauthenticated session passes every probe and delivers nothing, which is
  strictly worse than no session at all.
- **Each button press spawns a `gimx.exe` process (~250ms).** Fine for menus,
  too slow for frame-accurate input.

---

## Extending

**Add a tool** — write `provide() -> list[ToolSpec]` in a module under `tools/`,
then grant it to an agent by name or tag in `agents.yaml`.

**Add an agent** — subclass `BaseAgent`, implement `run(state) -> dict`, add an
entry to `agents.yaml` and a node to `graph.yaml`.

**Replace an agent** — point its `impl` at your class. A stricter verifier or a
console-specific planner needs no changes to the framework.

**Add a scenario** — drop a YAML file in `scenarios/`, or just type a sentence.

---

## Troubleshooting: black / blank capture

Run the diagnostic. It probes every video device and saves a frame from each,
so you can look rather than guess:

```bash
python ../temp_code_test/agentic/diagnose_capture.py
```

Reading the numbers (measured on this rig):

| std | tones | size | Meaning |
|---|---|---|---|
| ~35-49 | 255 | 1920x1080 | Real console content |
| 0.00 | 1-2 | any | True no-signal: console asleep, wrong HDMI port, or HDCP |
| 1-4 | 10-30 | 1280x720 | **A webcam in a dim room** — the wrong device |

That last row is the trap, and it has already bitten this project. A laptop
webcam is a *plausible* capture device: it opens, it returns frames, and in a
dim room those frames are dark — so every naive check agrees the console must
be off. OpenCV indices are positional and shift when USB devices change, so the
card moved from index 1 to index 0 and the framework quietly read the webcam.

**The fix, already applied:** `capture.py` now resolves the device by NAME via
ffmpeg (`auto_detect_device: true` in controls.yaml) and only falls back to
`opencv_index` if the name cannot be matched. The health agent additionally
treats a frame whose resolution does not match the configured size as
**blocking** — the wrong camera produces confident nonsense, which is worse
than an obvious failure. Smoke test 10 guards the mechanism.

---

## OCR

Text criteria (`text_present`, `no_error_dialog`) need OCR. Check what is
usable on this machine:

```bash
python _check_ocr.py        # tests both engines against a real captured frame
```

Two engines are tried in the order set by `verification.ocr.engines`:

| Engine | Install | Notes |
|---|---|---|
| `pytesseract` | `winget install UB-Mannheim.TesseractOCR` + `pip install pytesseract pillow` | Fast. Needs the separate **binary** — the pip package alone silently fails. |
| `paddleocr` | `pip install paddleocr paddlepaddle` | No binary, better on stylised fonts, ~200MB model on first use. |

An engine that imports but reads **nothing** counts as a failure and the next
is tried — otherwise "no text found" would be reported as a successful read of
an empty screen.

The executor OCRs the after-frame of **every** verified step, not only when a
plan asks for it. The verifier often needs to answer "was an error dialog
showing?" about a step nobody thought to OCR at planning time, and the frame is
already on disk.

Verified on this rig: tesseract reads the Xbox game-details page cleanly
(title, publisher, achievements, ESRB rating).

---

## Development

```bash
python ../temp_code_test/agentic/smoke_test.py       # 11 wiring checks, no hardware needed
python ../temp_code_test/agentic/test_fixes.py       # 6 regression tests for bugs found in live runs
python _check_ocr.py        # is OCR usable? read a real frame with each engine
python ../temp_code_test/agentic/diagnose_capture.py # probe every video device, save a frame from each
python ../temp_code_test/agentic/inspect_run.py      # summarise the latest run: plan, steps, verdict
python ../temp_code_test/agentic/verify_report.py    # confirm the report's screenshots actually resolve
```

`_test_fixes.py` guards three bugs that only appeared against real hardware:

1. **A wrong argument name aborted the run.** The planner wrote
   `wait_for_stable_screen(stability_duration=...)`; the real parameter is
   `settle`. A 6-step plan died at step 2 over a synonym. Tools are now
   argument-tolerant (aliases repaired, unknowns reported, missing-required
   still fails), tool signatures are exposed to the planner, and only
   hardware-level errors abort a run.

2. **Report screenshots were invisible.** Links were relative to the run
   directory, but `report.md` lives in `reports/`, so `frames/x.png` resolved
   to `reports/frames/x.png`. Every image was broken.

3. **PaddleOCR silently produced nothing.** It rejected `show_log`, a keyword
   removed upstream. Several constructor spellings are now tried in turn.

The routers in `graph/routing.py` are pure functions of the state, so the entire
control flow is testable with plain dictionaries — worth having, given how
expensive reproducing a hardware state is.

### One run at a time

The capture card can only be held by one process. Two overlapping runs will
both report missing frames, and the RCA agent will (correctly) call it a rig
fault. Let a run finish before starting the next.
