"""
report_tools.py - the report writers.

WHAT WAS DELETED, AND WHY
-------------------------
This module used to emit four formats of a step-by-step run record - json,
markdown, html and junit - with ~800 lines of template code behind them. All
of that has been REMOVED. The framework now produces exactly one report, the
GAME MECHANICS report in gameplay/mechanics_report.py, which answers the
question actually being asked of it:

    "which game mechanics were proven to work, and with what evidence?"

rather than "here is everything that happened, in order".

WHAT THIS COSTS, STATED PLAINLY
-------------------------------
There is no longer a junit.xml, so CI cannot consume a run directly. That was
a deliberate instruction, not an oversight. If CI integration is wanted again,
mechanics.json carries a per-mechanic verdict and is the natural thing to
translate - one small writer, not the old four.

WHY list_artifacts SURVIVED
---------------------------
It is tagged `analysis` as well as `report`, and the RCA/analysis agents use
it to enumerate the evidence they reason over. It writes nothing and formats
nothing, so the change of report format does not affect it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from registry import ToolContext, ToolSpec, fail, make_tool, ok


def _write_mechanics_report(ctx: ToolContext) -> Any:
    def run(session_dir: str = "", use_ai: bool = True,
            max_pairs: int | None = None) -> dict[str, Any]:
        """Build the game-mechanics report for a gameplay session."""
        # gameplay/ is on sys.path when launched via console.py, but the tool
        # loader only adds core/ and tools/ - so import by package path.
        import sys
        _root = Path(__file__).resolve().parent.parent
        if str(_root) not in sys.path:
            sys.path.insert(0, str(_root))
        from gameplay.mechanics_report import build_report

        target = Path(session_dir) if session_dir else ctx.artifacts.run_dir
        if not Path(target).is_dir():
            return fail(f"Not a session directory: {target}")

        try:
            written = build_report(target, settings=ctx.settings,
                                   use_ai=bool(use_ai), max_pairs=max_pairs)
        except Exception as exc:
            return fail(f"Mechanics report failed: {exc}")

        if not written:
            return fail("No report file could be written.")
        return ok(session_dir=str(target), written=written,
                  formats=sorted(written))

    return make_tool(
        run, "write_mechanics_report",
        "Build the game-mechanics report for a gameplay session: a per-"
        "mechanic PASS / FAILED / NOT TESTED / INCONCLUSIVE verdict backed "
        "by before-and-after screenshots, the measured frame delta and an AI "
        "audit of each frame pair. Writes markdown, html and json.")


def _list_artifacts(ctx: ToolContext) -> Any:
    def run() -> dict[str, Any]:
        return ok(run_dir=str(ctx.artifacts.run_dir),
                  files=ctx.artifacts.files,
                  frames=ctx.artifacts.list_frames())

    return make_tool(
        run, "list_artifacts",
        "List every file produced during this run: frames, logs and reports.")


def provide() -> list[ToolSpec]:
    return [
        ToolSpec("write_mechanics_report",
                 "Write the game-mechanics report with screenshots and AI "
                 "diagnostics.",
                 ["report"], _write_mechanics_report),
        ToolSpec("list_artifacts", "List all files produced this run.",
                 ["report", "analysis"], _list_artifacts),
    ]
