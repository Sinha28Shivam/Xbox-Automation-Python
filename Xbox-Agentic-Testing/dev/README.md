# dev/ - developer tools (not part of the product)

Hand-run scripts used while building and calibrating the framework. They are
not imported by the framework, not run in CI, and several **send real input
to the console**. Read a script before running it.

Run from `Xbox-Agentic-Testing/`, e.g. `python dev\probes\peek_small.py`.
Output images go to `scratch/` (git-ignored).

| Folder | Contents | Hardware |
|---|---|---|
| `probes/` | One-off rig probes: capture a frame (`peek_small`, `_now`), test a button or stick (`hw_pad`, `hw_guide`, `hw_stick_down`, `hw_b`, `probe_rs_emote`, `locomotion_probe`, `verify_dodge_fix`), inspect saved frames (`check_*`, `inspect_*`, `crop_menu`, `probe_heart*`) | `hw_*`, `probe_rs_emote`, `locomotion_probe`, `verify_dodge_fix`, `peek_small`, `_now` touch the rig |
| `runners/` | Thin wrappers that start one gameplay loop from the current game state (`run_dungeons`, `run_dungeons_fast`, `run_minecraft_gameplay`, `run_stage`, `run_village_loot`) | Yes - they play the game |
| `calibration/` | `dungeons_health_calib`: checks the pixel heart-gauge reader against saved run frames | No |

The supported entry point for running tests is `xbox-agentic` /
`python console.py`. Regression tests live in `tests/`.
