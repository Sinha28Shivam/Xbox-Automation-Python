# Onboarding: where to change what

Read this first. Then follow the guide for your task:

- **New test of a console screen/menu** (Guide, dashboard, Store, Settings):
  [ADDING_SCENARIOS.md](ADDING_SCENARIOS.md)
- **New game to play, or changing how a game is played**:
  [ADDING_GAMEPLAY.md](ADDING_GAMEPLAY.md)
- **Running tests and reading results**: [OPERATIONS.md](OPERATIONS.md)

## 1. Set up (once)

```powershell
cd Xbox-Agentic-Testing
python -m venv ..\venv ; ..\venv\Scripts\Activate.ps1
pip install -r requirements.lock
pip install -e ".[dev]"
copy .env.example .env          # put your ANTHROPIC_API_KEY in it
pre-commit install              # run lint/type checks on every commit
python -m pytest                # must pass before you change anything
```

`xbox-agentic ...` and `python console.py ...` are the same command.

## 2. How a run works

```
scenario YAML
   |
health ──> scenario_validator ──> planner ──> executor ──> verifier ──> reporter
  (rig ok?)   (YAML -> criteria)   (steps)    (presses,     (PASS only    (HTML +
                                    ^          captures,     with screen   JSON)
                                    |          OCR)          evidence)
                                    └──── rca + lessons <── on failure ┘
```

Every agent, tool, prompt and threshold is configured in YAML. Most changes
are a config edit, not a code change.

## 3. Where to change what

| I want to... | Change this | Code change? |
|---|---|---|
| Add a test of a menu/screen | new `scenarios/<id>.yaml` | No |
| Change what a test checks | `success_criteria` in the scenario YAML | No |
| Change how a menu is navigated | `navigation_guidance` in the scenario YAML | No |
| Change a button name/alias | `../Xbox-Automation-Python/config/controls.yaml` | No |
| Change a timeout / step limit / replan count | `config/settings.yaml` (`runtime:`) or scenario `timeout_seconds` | No |
| Change OCR engine or thresholds | `config/settings.yaml` (`verification:`) | No |
| Teach the guard a new game's screen text | `config/screen_signatures.yaml` (`games:`) | No |
| Change learning behaviour | `config/settings.yaml` (`learning:`) | No |
| Change what an agent is told | `config/prompts/<agent>.j2` | No (prompt) |
| Swap the LLM provider/model | `.env` (`AGENTIC_LLM_PROVIDER`, `*_MODEL`) | No |
| Add a new kind of controller action or screen check | `tools/*.py` (new tool) | Yes |
| Add a new game to play autonomously | `tools/game_profiles/` + `tools/<game>_gameplay.py` | Yes |
| Change a verdict rule | `agents/verifier_agent.py`, `core/schemas.py` | Yes (careful) |
| Change the report layout | `core/professional_report.py`, `core/scenario_report.py` | Yes |
| Add a runtime safety rule from a lesson | `core/runtime_lessons.py` | Yes |

## 4. Code map

| Folder | What lives there |
|---|---|
| `agents/` | One file per agent (`planner_agent.py`, `executor_agent.py`, ...). `base.py` = shared LLM/tool plumbing |
| `core/` | Schemas (`schemas.py` - the data contracts), config, artifacts, lessons, reports, logging |
| `tools/` | Everything an agent can call: input (`input_tools.py`), vision/OCR (`vision_tools.py`), navigation, per-game gameplay |
| `tools/game_profiles/` | One `GameProfile` per game: its move schema, prompt, success rule |
| `graph/` | Workflow wiring (`builder.py`, `routing.py`) and `runner.py` (runs one scenario) |
| `gameplay/` | The older `play` command (Max) and its mechanics report |
| `config/` | All YAML: settings, agents, graph, prompts, per-game controls, screen signatures |
| `scenarios/` | Test definitions |
| `tests/` | Offline tests (no hardware). `tests/offline/` = script-style regression tests |
| `dev/` | Hand-run probes - several press real buttons, read `dev/README.md` |
| `artifacts/` | Run output (git-ignored): `runs/<id>/`, `lessons/`, `routes/` |

## 5. Rules that keep the framework honest

1. **A PASS needs screen evidence.** "The button was sent" is never proof.
   Do not weaken `VerificationResult._no_pass_without_proof` in `core/schemas.py`.
2. **BLOCKED is not FAIL.** A broken rig must never be reported as a console bug.
3. **No blind input.** Confirm focus before pressing A; keep the in-game guards.
4. **Never hardcode a game or screen** in Python if YAML can describe it.
5. **Keep scenario `id`s stable**: lessons and cached routes are stored under them.

## 6. Before you open a PR

```powershell
ruff check . ; mypy ; python -m pytest        # CI runs exactly these
xbox-agentic run --file <your scenario> --dry-run
```

Then at least one real-hardware run of anything you touched, and a line in
`CHANGELOG.md`. Add a test for any bug you fix - `tests/test_replay_real_runs.py`
shows how to test against a real run's saved JSON.
