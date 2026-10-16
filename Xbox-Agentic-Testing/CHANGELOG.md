# Changelog

All notable changes to Xbox Agentic Testing. Format: [Keep a Changelog],
versions follow [Semantic Versioning].

[Keep a Changelog]: https://keepachangelog.com/en/1.1.0/
[Semantic Versioning]: https://semver.org/

## [Unreleased]

### Added
- Faster Dungeons combat (`tools/game_profiles/dungeons_speed.py`, settings
  `gameplay.dungeons`, every change has its own flag). Baseline
  dungeons-20261016-114012: 5.56 s per combat decision and 0.94 attacks per
  combat cycle; the hero was idle about 4 s of every 5.5 s.
  - The hero keeps attacking while the model thinks, up to 12 s. It checks
    the heart gauge between bursts and breaks off to drink or roll. In the
    offline fake rig (2.5 s model delay, 8 cycles) this went from 7 to 31
    volleys.
  - While engaged, the model gets a 9-field combat schema and a short prompt,
    with a full reading every 3rd cycle.
  - Frames sent to the model are 640 px while engaged.
  - Combat timing: 1.2 s volley, 6-swing burst, roll after every 2nd burst,
    artifacts on an 8 s cooldown.
  - The report gains a Combat speed card, and each cycle in the timeline
    shows decision mode, decision time and the attacks made while thinking.
- Optional engine hooks `GameProfile.sustain`, `fast_frame_model`,
  `vision_size` and `encode_frame(width, quality)`. Max and Minecraft are
  unchanged.
- Learning for Dungeons gameplay (`core/gameplay_lessons.py`,
  `tools/game_profiles/dungeons_learning.py`). Lessons for deaths, skipped
  chests/items, and actions with no visible result are recorded from the cycle
  log. They are applied at the next run's start: earlier potion, interaction
  priority, and a prompt hint. Resolved on PASS, expire after 3 quiet runs.
- The Dungeons report now uses the professional dashboard properly: real
  pass/fail counts, a run summary, cycle links to frames, a Learning section,
  key moments with frames, and a per-cycle timeline.
  `dungeons-scenarios.html` is now the same dashboard; the old table is in
  `dungeons-scenarios-raw.html`.

### Fixed
- Fixes from Dungeons run dungeons-20261016-133419:
  - Story captions are detected: the model said in_mission while the HUD was hidden and the heart read 0. They are now advanced with A instead of walked through (cycles 7-14 were wasted).
  - Chests, coins, villagers, gates and portals get x/y positions, and their distance is measured from them. A is pressed only within reach (frame 058 had 'very_close' chests about 3 hero-widths away). A chest approach is now finished instead of handed back to the navigator after one cycle.
  - 'Fighting mobs' can now pass for ranged fights: the next frame shows fewer enemies on screen and no health lost.
  - No 'Saving villagers' lesson is recorded for a calm villager. False lessons G006 and G007 were resolved by hand.
  - The death lesson stays mild (80% / -18%) unless the last run had 2+ deaths. Previously it drank at 90% and broke off attacks for a 10% hit.
  - A new objective stage needs 2 of the next 3 readings to agree, which removes misread stages.
- Dungeons report showed Passed 0 / Total 9 for a 4-PASS run (counts were
  never passed to the dashboard).
- 'Objective reached' was a false PASS: "Defeat the Illagers from the Rift"
  normalised to "defeat" and counted as a new stage
  (dungeons-20261016-114012, c42).

## [0.1.0] - 2026-10-15

First versioned release. Verified on hardware: `entry-point-game-activity`
PASS (run-20261015-125710, 41/41 steps, 241 s).

### Added
- Multi-agent LangGraph pipeline: health, scenario validator, planner,
  executor, verifier, RCA, reporter, recovery. Evidence-based verdicts (no
  PASS without observed proof); BLOCKED separate from FAIL.
- OCR-guided menu navigation (`guided_navigate_to_text`) with focus proof
  and already-open-page detection.
- Failure learning: lessons per scenario, shared across UI-surface tags,
  enforced at plan time and at runtime; expiry and `lessons` CLI.
- Professional HTML execution report, per-step timing, cross-run
  `trends` page.
- Screen signatures in `config/screen_signatures.yaml`; OCR-tolerant matching.
- Installable package with `xbox-agentic` command; pinned `requirements.lock`.
- Quality gates: ruff, mypy (core/agents), pre-commit, CI with coverage.
- Per-run `logs/run.log` and persistent `artifacts/agentic.log`; `--log-level`.
- Test suite: offline regression scripts, replay of real runs, scenario
  YAML validation, multi-process concurrency tests.

### Fixed
- Concurrent runs lost lesson updates (23/30 in a 6-process test); stores
  are now file-locked. Run ids no longer collide within one second.
- `minecraft-launch-requirement.yaml` shared an id with `minecraft-launch.yaml`.
- RCA no longer reports product defects without console evidence.
- 34 exception handlers that swallowed errors silently now log the cause.

### Changed
- Account-prompt guard bypass is now `account_prompt_policy` (default `warn`,
  unchanged behaviour).
- Hand-run probes moved from `scratch/` to `dev/`; `scratch/` is git-ignored.
