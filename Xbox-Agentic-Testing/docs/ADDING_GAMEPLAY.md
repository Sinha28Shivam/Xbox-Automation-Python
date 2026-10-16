# Adding or changing gameplay (the AI plays a game)

Gameplay = a closed loop: **capture frame -> vision model decides -> press
buttons -> capture again**, until the game shows success or the run stops.
This needs Python. For menu/screen tests use
[ADDING_SCENARIOS.md](ADDING_SCENARIOS.md) instead.

Reference implementation to copy: **Minecraft Dungeons**
(`tools/game_profiles/dungeons_profile.py`, `tools/dungeons_gameplay.py`,
`config/dungeons_controls.yaml`, `scenarios/dungeons-mission-inplay.yaml`).

## How the pieces fit

```
scenario YAML (tags: [<game>, gameplay], stage closed_loop_play)
   -> planner: capture_frame -> vision_guided_<game>_gameplay -> capture_frame
        -> tools/<game>_gameplay.py      registered tool, run options
            -> gameplay_engine.run_gameplay_loop(ctx, PROFILE, goal, max_cycles)
                -> tools/game_profiles/<game>_profile.py   PROFILE = GameProfile(...)
                     move schema, frame schema, prompt, execute_move,
                     success rule, reflexes, stuck rule
                -> config/<game>_controls.yaml   game action -> controller button
```

`tools/gameplay_engine.py` (shared loop) normally needs **no** change.

## Part A - Change how an existing game plays

| Change | Where |
|---|---|
| A button binding | `config/dungeons_controls.yaml` / `config/minecraft_controls.yaml` |
| What the AI is told to do | `SYSTEM_PROMPT` in `tools/game_profiles/<game>_profile.py` |
| Reflexes that must not wait for the AI (drink potion, dodge) | `_forced_moves` in the profile |
| How long a move/attack is held | constants at the top of the profile (e.g. `_TAP_HOLD`, `_FIGHT_SWINGS`) |
| HUD reads that override the AI (health from pixels) | `annotate_decision` in the profile |
| When "success" is accepted | `success_states` + `confirm_success` in the profile |
| Cycle budget for a run | scenario step `max_cycles`, or `--cycles` on the CLI |

Then:
```powershell
python -m pytest            # includes tests/offline/test_dungeons_*.py
xbox-agentic dungeons --cycles 20        # direct loop, game already in a mission
```

## Part B - Add a new game

### 1. Record the controls

Open the game's Controls screen and write `config/<game>_controls.yaml`
(copy `dungeons_controls.yaml`):

```yaml
meta: { source: "in-game Controls screen", verified: false }
bindings:
  jump:     { button: "a",  kind: "button",  label: "Jump" }
  attack:   { button: "rt", kind: "trigger", label: "Attack" }
```

Every `button` must exist in `../Xbox-Automation-Python/config/controls.yaml`.
Set `verified: true` only after pressing each one on hardware.

Add a small loader like `tools/dungeons_controls.py` (`button_for(action)`).

### 2. Write the profile - `tools/game_profiles/<game>_profile.py`

Required parts (see the `GameProfile` dataclass in `tools/gameplay_engine.py`):

| Part | What it is |
|---|---|
| `Move` model (pydantic) | One controller action the AI may choose, with a `Literal[...]` list of actions - the AI cannot pick anything else |
| `FrameDecision` model | What the AI returns per frame: what it sees, `scene_state`, `moves: list[Move]`, reasoning |
| `SYSTEM_PROMPT` | Rules of the game, what success looks like, what each action does |
| `execute_move(ctx, move)` | Turns one `Move` into button presses via the pad |
| `success_states` / `terminal_flag_key` / `build_terminal_evidence` | Which `scene_state` ends the run and what evidence is saved |

Optional hooks (defaults are safe): `forced_moves`, `fallback_moves`,
`annotate_decision`, `confirm_success`, `update_stuck_counters`,
`execute_moves` (several buttons at once), `use_skill_memory`.

End the file with:
```python
PROFILE = GameProfile(key="<game>", move_model=Move, frame_model=FrameDecision,
                      system_prompt=SYSTEM_PROMPT, default_goal="...",
                      artifact_prefix="<game>", json_artifact_name="<game>-cycles.json",
                      execute_move=execute_move, success_states={"level_complete"},
                      terminal_flag_key="level_complete",
                      terminal_evidence_key="level_evidence",
                      build_terminal_evidence=_terminal_evidence)
```

### 3. Register the tool - `tools/<game>_gameplay.py`

Copy `tools/dungeons_gameplay.py`. It must define `provide()` returning a
`ToolSpec(name="vision_guided_<game>_gameplay", tags=["input","vision","game"],
mutates_hardware=True, ...)` whose tool calls
`run_gameplay_loop(ctx, PROFILE, goal=goal, max_cycles=max_cycles)`.

Then add the module to the list in `build_default_registry`
(`tools/registry.py`), and the tool name to `_TOOL_ACTIONS` in
`agents/scenario_agent.py` so scenarios may use it.

### 4. Teach the safety guard the game's screen

In `config/screen_signatures.yaml` under `games:` add text that only appears
while the game runs (title, HUD words). Without it the guard treats the game
as the dashboard and aborts the run.

### 5. Planner bootstrap

In-gameplay scenarios get a fixed 3-step plan (`_fallback_in_gameplay_plan`
in `agents/planner_agent.py`). It currently routes Dungeons by tag
(`_is_dungeons`) and everything else to Max's `vision_guided_gameplay`.
Add a branch for your game's tag, modelled on `_fallback_dungeons_plan`.

### 6. Scenario - `scenarios/<game>-inplay.yaml`

Copy `scenarios/dungeons-mission-inplay.yaml`. Keep:
- `stages: [ { id: closed_loop_play, allowed_actions: [capture_frame, read_screen_text, vision_guided_<game>_gameplay] } ]`
- tags: the game name + `gameplay` (**never** `guide`/`dashboard`)
- precondition: "the game is already running and in a level" - the loop plays,
  it does not launch the game
- `timeout_seconds` large enough for the cycles (30-60 min is normal)

### 7. Tests (no hardware)

Copy `tests/offline/test_dungeons_offline.py`: assert every `Move.action`
resolves to a real control, `execute_move` presses the expected buttons on a
fake pad, and `forced_moves` fire when they should. Then:

```powershell
ruff check . ; mypy ; python -m pytest
xbox-agentic run --file scenarios\<game>-inplay.yaml --dry-run
```

### 8. Hardware, in this order

1. Press each binding with a `dev/probes/` script; set `verified: true`.
2. Short loop: start the game in a level, run 10-20 cycles directly.
3. Full scenario run; read `execution-report.html` and the cycle log
   (`<game>-cycles.json`) frame by frame before trusting a PASS.

### 9. Learning (optional)

Copy `tools/game_profiles/dungeons_learning.py`. Implement `detect` (what
the cycle log PROVES went wrong), `assess` (fixed / quiet / untested),
`adjustments` (bounded counter overrides your `forced_moves` read) and
`prompt_text`. Call `core/gameplay_lessons.py` `open_lessons` at the start and
`update` at the end, as `tools/dungeons_gameplay.py` does. Lessons appear in
`xbox-agentic lessons list gameplay-<game>`. Tests:
`tests/test_dungeons_learning.py`.

### 10. Speed hooks (optional)

`GameProfile` has optional hooks for fast games. They default to off, and are
used in `tools/game_profiles/dungeons_speed.py`:

- `sustain` + `merge_sustain`: keep acting on the input thread while the
  model thinks. It needs `overlap_act=True` and must return soon after its
  stop event is set.
- `fast_frame_model` + `fast_system_prompt` + `use_fast_decision` +
  `to_full_decision`: a smaller schema for hot situations.
- `vision_size`: the frame width and JPEG quality sent to the model.

Tests: `tests/test_dungeons_speed.py`.

## Rules for gameplay code

- **Only the `Literal` actions in the Move model can ever be pressed.** Keep
  that list tight; it is the safety boundary for what the AI controls.
- **Success must be confirmed, not claimed.** Implement `confirm_success`
  with something observable (screen text, HUD pixels).
- **Reflexes go in `forced_moves`**, not the prompt: a ~5 s AI round-trip is
  too slow to save the character.
- **Stuck detection stays on.** If nothing on screen changes, the loop must stop.
- Measure, don't guess: hold times and thresholds in the existing profiles are
  annotated with the hardware run that justified them - do the same.

## Checklist

- [ ] `config/<game>_controls.yaml`, all buttons pressed on hardware
- [ ] Profile with Move/FrameDecision models, prompt, `execute_move`, success rule, `PROFILE`
- [ ] `tools/<game>_gameplay.py` with `provide()`; added to `registry.py` and `_TOOL_ACTIONS`
- [ ] Game signatures in `config/screen_signatures.yaml`
- [ ] Planner bootstrap branch for the game's tag
- [ ] In-play scenario YAML; offline tests; ruff + mypy + pytest pass
- [ ] Short loop, then full run, reviewed frame by frame
