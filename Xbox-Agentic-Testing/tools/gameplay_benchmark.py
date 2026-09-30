"""gameplay_benchmark.py - a persistent, (profile, scenario_name)-keyed
journal of gameplay-loop RUN RESULTS, so "did today's change make this
better or worse than last time" has an actual answer instead of needing a
human to re-read terminal logs.

WHY THIS EXISTS
----------------
`run_gameplay_loop` already returns honest per-cycle evidence (frame paths,
measured pixel deltas) - but nothing PERSISTS that evidence across separate
runs in a comparable way. Today, checking whether a code change (say,
turning on skill_memory or the curriculum wrapper) actually helped means
re-reading two different terminal logs by eye. This module does for RUN
OUTCOMES what location_memory.py did for PLACES and skill_memory.py did for
MOVE SEQUENCES: a small persisted journal, same cross-run pattern, so a
scenario replayed later can be compared numerically against its own history.

WHAT COUNTS AS "BETTER"
--------------------------
Deliberately narrow and honest, matching the loop's own rule that a pixel
delta is evidence of SOMETHING changing, not proof of the RIGHT thing
happening:
  - cycles_to_terminal: cycles until profile.success_states was reached, or
    None if the run ended without ever reaching it (max_cycles/stuck/error).
    Lower is better, and None is explicitly WORSE than any finite number,
    never silently sorted as equal to it.
  - mean_delta / max_delta: carried through unchanged from the loop's own
    honest evidence - this module does not reinterpret them.
  - reached_terminal: whether success_states was ever observed at all -
    the most basic signal, checked before the cycle count is trusted.
No score is invented that the loop itself did not already compute.

PERSISTENCE ACROSS RUNS
------------------------
Same top-level artifacts/ location as world_memory.json and
skill_memory.json - a benchmark run recorded today must be comparable
against one recorded last week, in a later session.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from registry import ToolContext, ToolSpec, fail, make_tool, ok
from gameplay_engine import GameProfile, run_gameplay_loop


def _memory_path(ctx: ToolContext) -> Path:
    run_dir = ctx.artifacts.run_dir
    base_dir = run_dir.parent.parent if "runs" in run_dir.parts else run_dir
    return base_dir / "benchmark_memory.json"


def _load(ctx: ToolContext) -> list[dict[str, Any]]:
    path = _memory_path(ctx)
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _save(ctx: ToolContext, entries: list[dict[str, Any]]) -> None:
    path = _memory_path(ctx)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")


# ===========================================================================
# Registry of runnable profiles, by key - a benchmark names a profile by
# string (e.g. "minecraft") so results can be recorded/compared without the
# caller needing to import the profile module directly.
# ===========================================================================
def _resolve_profile(profile_key: str) -> GameProfile:
    if profile_key == "minecraft":
        from game_profiles.minecraft_profile import PROFILE
        return PROFILE
    if profile_key == "max":
        from game_profiles.max_profile import PROFILE
        return PROFILE
    raise KeyError(f"Unknown profile '{profile_key}'. Known: minecraft, max")


# ===========================================================================
# Run + record a benchmark
# ===========================================================================
def run_benchmark_impl(
    ctx: ToolContext,
    profile: str,
    scenario_name: str,
    goal: str,
    max_cycles: int = 10,
    notes: str = "",
) -> dict[str, Any]:
    """Run `profile`'s gameplay loop toward `goal` for up to `max_cycles`,
    score the result, and append it to this scenario's run history.

    Scoring is derived ONLY from what run_gameplay_loop itself already
    measured (cycles_run, terminal flag, mean/max delta) - see the module
    docstring's "what counts as better" note.
    """
    try:
        game_profile = _resolve_profile(profile)
    except KeyError as exc:
        return fail(str(exc), profile=profile, scenario_name=scenario_name)

    loop_result = run_gameplay_loop(ctx, game_profile, goal=goal,
                                    max_cycles=max_cycles)
    if not loop_result.get("ok"):
        return fail(f"Gameplay loop did not complete: {loop_result.get('error')}",
                    profile=profile, scenario_name=scenario_name)

    reached_terminal = bool(loop_result.get(game_profile.terminal_flag_key))
    cycles_to_terminal = loop_result.get("cycles_run") if reached_terminal else None

    record = {
        "profile": profile,
        "scenario_name": scenario_name,
        "goal": goal,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "max_cycles": max_cycles,
        "cycles_run": loop_result.get("cycles_run"),
        "reached_terminal": reached_terminal,
        "cycles_to_terminal": cycles_to_terminal,
        "mean_delta": loop_result.get("mean_delta"),
        "max_delta": loop_result.get("max_delta"),
        "stop_reason": loop_result.get("stop_reason"),
        "notes": notes,
    }

    entries = _load(ctx)
    entries.append(record)
    _save(ctx, entries)

    history = [e for e in entries if e["profile"] == profile
              and e["scenario_name"] == scenario_name]

    return ok(record=record, total_runs_for_scenario=len(history),
             journal_path=str(_memory_path(ctx)))


def _run_benchmark(ctx: ToolContext) -> Any:
    def run(profile: str, scenario_name: str, goal: str,
            max_cycles: int = 10, notes: str = "") -> dict[str, Any]:
        return run_benchmark_impl(ctx, profile=profile, scenario_name=scenario_name,
                                  goal=goal, max_cycles=max_cycles, notes=notes)

    return make_tool(
        run, "run_benchmark",
        "Run a game profile's gameplay loop toward a goal, score the "
        "result (cycles-to-success, reached-terminal, mean/max pixel "
        "delta - all taken directly from the loop's own measured "
        "evidence), and record it under a named scenario for later "
        "comparison against its own history.")


# ===========================================================================
# Compare a scenario's run history
# ===========================================================================
def compare_benchmark_runs_impl(ctx: ToolContext, profile: str,
                                scenario_name: str) -> dict[str, Any]:
    """Every recorded run for this (profile, scenario_name), oldest first,
    plus a best-so-far summary. Does not judge "improving" or "regressing" -
    that comparison is left to whoever reads the list, same honesty rule as
    the gameplay loop itself.
    """
    entries = _load(ctx)
    history = [e for e in entries if e["profile"] == profile
              and e["scenario_name"] == scenario_name]
    if not history:
        return fail(f"No recorded runs for scenario '{scenario_name}' "
                    f"under profile '{profile}' yet.",
                    profile=profile, scenario_name=scenario_name)

    finite = [e["cycles_to_terminal"] for e in history
             if e["cycles_to_terminal"] is not None]
    best_cycles_to_terminal = min(finite) if finite else None
    terminal_rate = round(
        sum(1 for e in history if e["reached_terminal"]) / len(history), 3)

    return ok(profile=profile, scenario_name=scenario_name,
             runs=history, total_runs=len(history),
             best_cycles_to_terminal=best_cycles_to_terminal,
             terminal_rate=terminal_rate,
             journal_path=str(_memory_path(ctx)))


def _compare_benchmark_runs(ctx: ToolContext) -> Any:
    def run(profile: str, scenario_name: str) -> dict[str, Any]:
        return compare_benchmark_runs_impl(ctx, profile=profile,
                                           scenario_name=scenario_name)

    return make_tool(
        run, "compare_benchmark_runs",
        "List every recorded benchmark run for a named scenario, oldest "
        "first, plus the best cycles-to-terminal seen so far and the "
        "fraction of runs that ever reached the terminal/success state. "
        "Does not itself judge improvement or regression.")


def provide() -> list[ToolSpec]:
    return [
        ToolSpec(name="run_benchmark",
                 description="Run a gameplay-loop scenario and record its score.",
                 tags=["input", "vision", "game", "memory"],
                 factory=_run_benchmark, mutates_hardware=True),
        ToolSpec(name="compare_benchmark_runs",
                 description="List a scenario's recorded runs and best result.",
                 tags=["analysis", "memory"],
                 factory=_compare_benchmark_runs, mutates_hardware=False),
    ]
