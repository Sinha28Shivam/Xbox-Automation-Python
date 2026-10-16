# Operations runbook

For whoever runs tests on the rig. Commands assume `Xbox-Agentic-Testing/`
and the venv installed as in the README (`xbox-agentic` == `python console.py`).

## 1. Before a session

1. Console on, signed in to the **test account**, on the dashboard, Guide closed.
2. Capture card connected; close RECentral or anything else holding it.
3. GIMX running and **authenticated**: hold the controller's Guide button 2 s.
4. Check the rig:
   ```powershell
   xbox-agentic health
   ```
   Exit `0` = ready. Exit `2` = BLOCKED: fix what it lists before going on.

## 2. Running a scenario

```powershell
xbox-agentic run --file scenarios\entry-point-game-activity.yaml --dry-run   # validate first
xbox-agentic run --file scenarios\entry-point-game-activity.yaml
```

| Exit code | Verdict | Meaning |
|---|---|---|
| 0 | PASS | Every required criterion observed on screen |
| 1 | FAIL | Product behaviour tested and not met (or automation defect; see RCA) |
| 2 | BLOCKED | Rig problem: nothing about the console was tested |
| 3 | INCONCLUSIVE | Not enough evidence (always the result of `--dry-run`) |
| 4 | ERROR | Framework crash - our bug, never a console failure |

## 3. Reading results

Each run: `artifacts/runs/<run-id>/`.

| What | Where |
|---|---|
| Dashboard report (open in a browser) | `reports/execution-report.html` |
| Everything shown on screen, timestamped | `logs/run.log` |
| Why it failed | report "Root cause analysis"; `reports/rca.json` |
| What it saw at each step | report "Step timeline" thumbnails; `frames/` |
| Lessons used / recorded | report "Failure learning" |

History across runs: `xbox-agentic trends` -> `artifacts/trends.html`.

## 4. Common problems

| Symptom | Likely cause | Action |
|---|---|---|
| BLOCKED, "frame is FLAT / no signal" | Console asleep, HDMI in wrong port, HDCP | Wake console; HDMI into the card's IN port; disable HDCP |
| BLOCKED, wrong resolution | Webcam opened instead of capture card | Check `capture` device name in `controls.yaml` |
| Steps dispatched but screen never changes | GIMX session not authenticated | Hold Guide 2 s; re-run health |
| Aborted "Blind tapping prevented" | Console on Store/account/dashboard during gameplay | Check start state; see `config/screen_signatures.yaml` |
| Aborted "Run timeout exceeded" | Scenario `timeout_seconds` too low | Raise it in the scenario YAML |
| Navigation "visible but never highlighted" | Target text is on screen but focus is elsewhere | Check `direction` / `require_focus` in `navigation_guidance` |
| Exit 4 FRAMEWORK ERROR | Bug in the framework | Re-run with `--traceback`; attach `logs/run.log` |

More detail: `--log-level DEBUG` also records every error a fallback
suppressed (logger `suppressed`).

## 5. Learning

```powershell
xbox-agentic lessons                                   # open lessons
xbox-agentic lessons list <scenario-id> --all
xbox-agentic lessons resolve <scenario-id> L002 --reason "fixed in code"
```

To compare against the old combat behaviour, turn the speed-ups off one at
a time in `config/settings.yaml` under `gameplay.dungeons`, or set
`DUNGEONS_SUSTAIN=false` or `DUNGEONS_FAST_SCHEMA=false`. The report's
**Combat speed** card shows the decision time and the attacks per combat cycle.

Dungeons runs keep their lessons in `gameplay-dungeons`. Each run lists them under
**Learning** in `reports/execution-report.html`, along with how many times
they changed play.

Lessons close on a PASS, after 3 runs without recurring, or by hand. Resolve
a lesson manually when the bug it describes was fixed in code.

## 6. Adding a scenario

Full guides: [ADDING_SCENARIOS.md](ADDING_SCENARIOS.md) (menus/screens),
[ADDING_GAMEPLAY.md](ADDING_GAMEPLAY.md) (games). Short version:

1. Copy an existing YAML in `scenarios/`; give it a new, stable `id:`.
2. Tag it (`guide`, `dashboard`, ... share lessons; game scenarios must not
   use those tags).
3. For menu navigation use `navigation_guidance` (`target_text`, `direction`,
   `max_presses`/`max_attempts`, `require_focus`).
4. `python -m pytest tests\test_scenario_files.py` - validates the file.
5. `xbox-agentic run --file <file> --dry-run`, then a real run.

New game: add its on-screen signatures under `games:` in
`config/screen_signatures.yaml`, or the in-game guard cannot recognise it.

## 7. Releasing a change

```powershell
ruff check . ; mypy ; python -m pytest       # same gates as CI
```

Then one real-hardware run of an affected scenario, and an entry in
`CHANGELOG.md`.
