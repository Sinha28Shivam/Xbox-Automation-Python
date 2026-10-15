# Changelog

All notable changes to Xbox Agentic Testing. Format: [Keep a Changelog],
versions follow [Semantic Versioning].

[Keep a Changelog]: https://keepachangelog.com/en/1.1.0/
[Semantic Versioning]: https://semver.org/

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
