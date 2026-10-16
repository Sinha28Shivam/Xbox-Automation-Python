# Adding a test scenario (menus, Guide, dashboard, Store, Settings)

A scenario is **one YAML file** in `scenarios/`. No Python is needed. Use this
guide for tests of console UI. To make the framework *play a game*, see
[ADDING_GAMEPLAY.md](ADDING_GAMEPLAY.md).

Working example to copy: `scenarios/entry-point-game-activity.yaml`
(passed on hardware, run-20261015-125710 and run-20261015-141407).

## Step 1 - Write down the manual test

Do it by hand on the console first and note:
- the **start screen** (e.g. "dashboard, Guide closed"),
- every button press and **what you see after it**,
- the exact on-screen text that proves each step worked (e.g. "Achievements"),
- the **end screen**.

If you cannot say what you would see on screen, the framework cannot check it.

## Step 2 - Create the file

`scenarios/<your-id>.yaml`:

```yaml
id: settings-network-check          # unique, never change it later
title: "Network settings open from the Guide"
description: >
  From the dashboard, open the Guide, go to Profile & system > Settings >
  General > Network settings and verify the network status is shown.
goal: >
  Open the Guide, navigate to Settings, open Network settings and verify the
  connection status is visible, then return to the dashboard.
console: xbox_one

preconditions:
  - The console is powered on and showing the Xbox dashboard
  - The Guide is closed
  - The GIMX session is authenticated
  - The capture card is connected and no other application is using it

success_criteria:
  - description: >
      The Settings screen opens from the Guide's Profile & system section.
    check_type: vision_judgement
    parameters:
      expected_screen: Xbox Settings
      navigation_guidance:            # optional - see Step 4
        target_text: Settings
        direction: down
        max_attempts: 8
        interval: 0.8
        require_focus: true
    required: true

  - description: Network settings shows the current connection status.
    check_type: text_present
    parameters:
      text: "Network settings"
    required: true

  - description: No error dialog appears during the journey.
    check_type: no_error_dialog
    required: true

tags:
  - guide          # shares lessons with other Guide tests
  - settings       # shares lessons with other Settings tests
  - network        # label only

timeout_seconds: 600
max_steps: 50
```

## Step 3 - Choose the success criteria

Each criterion must be something visible on screen.

| `check_type` | Use for | `parameters` | Judged by |
|---|---|---|---|
| `text_present` | Exact words must appear on screen | `text` | Code (OCR match) |
| `no_error_dialog` | No "Something went wrong", error codes, etc. | - | Code (OCR) |
| `screen_change` | The screen visibly changed | optional `min_delta` | Code (pixel diff) |
| `vision_judgement` | "This screen is the X page" | `expected_screen`, optional `required_visible: [...]` | AI verifier, from frames |
| `text_absent`, `screen_match` | Accepted, but no dedicated check yet | free-form | AI verifier, from frames |

Prefer the code-judged types where you can: they are deterministic and
cannot be argued with. Use `vision_judgement` for layout/screen identity.

Mark `required: true` on everything that must hold for a PASS.

## Step 4 - Menu navigation (`navigation_guidance`)

For "move through a menu until X", add `navigation_guidance` to the criterion
for that destination. The framework then presses one direction at a time,
reads the screen after each press, and stops when the target is reached -
instead of guessing a number of presses.

| Key | Meaning |
|---|---|
| `target_text` | Text of the item to reach (case/punctuation ignored) |
| `direction` | `up`, `down`, `left`, `right` |
| `max_presses` | Exact known number of presses (e.g. Game Activity = 3 Right) |
| `max_attempts` | Upper bound when the count is unknown |
| `interval` | Seconds to wait for the menu animation (0.8 works on the Guide) |
| `require_focus` | `true` (default): stop only when the item is **highlighted**. `false`: stop when the text is **visible** - use for tab headings like "Game Activity", which are never highlighted |

These values are applied exactly as written; the AI planner cannot change them.

## Step 5 - Tags

- `guide`, `dashboard`, `store`, `search`, `settings` = **shared tags**: lessons
  learned by one scenario with the tag help every other scenario with it.
- A tag `dashboard` or `guide` also tells the executor the dashboard is a valid
  screen (it will not abort with "blind tapping prevented").
- **Never** put these tags on a game scenario.

## Step 6 - Validate offline

```powershell
python -m pytest tests\test_scenario_files.py      # YAML, check types, unique id
xbox-agentic run --file scenarios\<your-id>.yaml --dry-run
```

The dry run must end `INCONCLUSIVE` (it never presses anything). Open the plan
in `artifacts\runs\<run-id>\reports\plan-r1.json` and check the steps make sense.

## Step 7 - Run on the console

```powershell
xbox-agentic health
xbox-agentic run --file scenarios\<your-id>.yaml
```

Open `artifacts\runs\<run-id>\reports\execution-report.html`.

## Step 8 - If it fails

| Report shows | Fix in the YAML |
|---|---|
| Guided navigation "target never appeared" | Wrong `direction`, or the item is in another section |
| "visible but never highlighted" | Use `require_focus: false` if it is a heading, or check `direction` |
| "Run timeout exceeded" | Raise `timeout_seconds` (each step costs ~5 s) |
| A criterion FAILED but the screen looks right | Make `description` / `expected_screen` match what is really shown |
| Wrong start screen | Fix `preconditions`; start the console from that screen |

Each failure is recorded as a **lesson** and given to the planner next run
(`xbox-agentic lessons`). After a PASS, lessons close and the passing plan is
replayed on later runs with no AI planning.

## Checklist

- [ ] Unique, stable `id`
- [ ] Every criterion is visible on screen; required ones marked `required: true`
- [ ] `navigation_guidance` for menu travel; `require_focus: false` on headings
- [ ] Correct tags (shared tags only for UI scenarios)
- [ ] `pytest tests\test_scenario_files.py` passes; dry run is INCONCLUSIVE
- [ ] One real PASS on the console before relying on it
