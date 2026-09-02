"""
console.py - the command line for the agentic console testing framework.

    python console.py run "Press A and confirm the screen changes"
    python console.py run --file scenarios/dashboard-navigation.yaml
    python console.py run "..." --dry-run
    python console.py health
    python console.py info
    python console.py tools
    python console.py interactive

DESIGN
------
Thin by intention. Everything real lives in TestRunner; this file parses
arguments, prints results and picks an exit code. Keeping it thin means the
framework is equally usable as a library, which matters for pytest integration
and for anyone embedding it in a larger harness.

EXIT CODES
----------
    0  pass
    1  fail          the console misbehaved
    2  blocked       the RIG was broken - we never tested the console
    3  inconclusive  we could not tell
    4  error         the framework itself broke

Blocked has its own code deliberately. A CI job that cannot distinguish "the
product is broken" from "the test rig is broken" will eventually ignore both.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
for _sub in ("core", "tools", "agents", "graph"):
    _path = str(_ROOT / _sub)
    if _path not in sys.path:
        sys.path.insert(0, _path)

from runner import TestRunner, verdict_exit_code       # noqa: E402
from schemas import Verdict                            # noqa: E402

_LINE = "=" * 72


# ===========================================================================
# Output
# ===========================================================================
def print_report(report) -> None:
    if report is None:
        print("\nNo report was produced - the run did not complete.")
        return

    print("\n" + _LINE)
    print(f"  {report.verdict.value.upper()}  -  {report.scenario_title}")
    print(_LINE)
    print(f"\n{report.summary}\n")

    if report.metrics:
        m = report.metrics
        print(f"  steps run        : {m.get('steps_run')}/{m.get('steps_planned')}")
        print(f"  screen changes   : {m.get('steps_with_screen_change')}")
        print(f"  mean delta       : {m.get('mean_screen_delta')}")
        print(f"  replans          : {m.get('replans')}")
        print(f"  duration         : {report.duration_seconds:.1f}s")

    if report.verification and report.verification.criteria:
        print("\n  Criteria:")
        for c in report.verification.criteria:
            print(f"    [{'x' if c.met else ' '}] {c.criterion}")

    # Printed to the terminal, not just written to the report. Someone reading
    # only the console output should still see the limits of the result.
    if report.caveats:
        print("\n  This result does NOT prove:")
        for caveat in report.caveats:
            print(f"    - {caveat}")

    if report.rca:
        print(f"\n  Root cause ({report.rca.failure_class.value}): "
              f"{report.rca.primary_cause}")
        for i, rec in enumerate(report.rca.recommendations[:3], 1):
            print(f"    {i}. {rec}")

    if report.report_files:
        print("\n  Reports:")
        for fmt, path in report.report_files.items():
            print(f"    {fmt:9s} {path}")
    print()


def print_health(runner: TestRunner) -> int:
    """Run only the health probes - no LLM, no API key needed."""
    from artifacts import ArtifactStore
    from base import AgentContext
    from health_agent import HealthAgent
    from llm import LLMFactory
    from prompts import PromptLibrary
    from registry import build_default_registry
    from state import initial_state

    artifacts = ArtifactStore(
        root=runner.settings.resolve_path("paths.artifacts_dir", "./artifacts"),
        run_id="health-check", enabled=False)
    tools = build_default_registry(runner.hardware, artifacts, runner.settings)

    context = AgentContext(
        settings=runner.settings,
        agents_config=runner.agents_config,
        llm_factory=LLMFactory(runner.settings),
        prompts=PromptLibrary(
            runner.settings.resolve_path("paths.prompts_dir", "./config/prompts")),
        tools=tools,
        artifacts=artifacts,
    )

    agent = HealthAgent(context, "health")
    result = agent(initial_state("health-check", "", "text", {}, "", False, ""))
    report = result.get("health")

    print("\n" + _LINE)
    print("  RIG HEALTH")
    print(_LINE)
    if report is None:
        print("\n  The health check itself failed:")
        for err in result.get("errors", []):
            print(f"    {err}")
        return 4

    for component in report.components:
        print(f"  [{'OK  ' if component.ok else 'FAIL'}] {component.name}")
        if component.detail:
            print(f"         {component.detail}")
        if component.remediation:
            print(f"         FIX: {component.remediation}")

    for issue in report.blocking_issues:
        print(f"\n  BLOCKING: {issue}")
    for warning in report.warnings:
        print(f"\n  WARNING: {warning}")

    print(f"\n{_LINE}")
    print(f"  {report.summary}")
    print(_LINE + "\n")
    return 0 if report.healthy else 2


# ===========================================================================
# Commands
# ===========================================================================
def cmd_run(args: argparse.Namespace) -> int:
    overrides = {}
    if args.dry_run:
        overrides["dry_run"] = True
    if args.max_steps:
        overrides["max_steps"] = args.max_steps
    if args.no_artifacts:
        overrides["save_frames"] = False

    with TestRunner(args.config_dir, overrides or None) as runner:
        if args.file and args.requirement_file:
            print("ERROR: use either --file or --requirement-file, not both")
            return 4

        if args.requirement_file:
            path = Path(args.requirement_file)
            if not path.is_file():
                print(f"ERROR: requirement file not found: {path}")
                return 4
            report = runner.run(str(path.resolve()), "requirement_file")
        elif args.file:
            path = Path(args.file)
            if not path.is_file():
                print(f"ERROR: scenario file not found: {path}")
                return 4
            report = runner.run(str(path.resolve()), "file")
        else:
            if not args.scenario:
                print("ERROR: provide a scenario, or use --file / --requirement-file")
                return 4
            report = runner.run(" ".join(args.scenario), "text")

        print_report(report)
        return verdict_exit_code(report)


def cmd_health(args: argparse.Namespace) -> int:
    with TestRunner(args.config_dir) as runner:
        return print_health(runner)


def cmd_info(args: argparse.Namespace) -> int:
    with TestRunner(args.config_dir) as runner:
        info = runner.describe()

    if args.json:
        print(json.dumps(info, indent=2, default=str))
        return 0

    print("\n" + _LINE)
    print("  FRAMEWORK CONFIGURATION")
    print(_LINE)
    print(f"\n  config dir  : {info['config_dir']}")
    print(f"  LLM         : {info['llm']['provider']} / {info['llm']['model']}")
    print(f"  route mode  : {info['route_mode']}")
    print(f"  dry run     : {info['runtime'].get('dry_run')}")

    print("\n  Hardware adapters:")
    for name, adapter in info["hardware"]["adapters"].items():
        mark = "OK  " if adapter["available"] else "FAIL"
        print(f"    [{mark}] {name:10s} {adapter['path']}")
        if adapter["error"]:
            print(f"             {adapter['error']}")

    print(f"\n  controls.yaml: {info['hardware']['controls_config']}")
    print(f"    exists: {info['hardware']['controls_exists']}")

    print("\n  Agents:")
    for role, spec in info["agents"].items():
        mark = "on " if spec["enabled"] else "off"
        print(f"    [{mark}] {role:20s} {spec['impl']}")
    print()
    return 0


def cmd_tools(args: argparse.Namespace) -> int:
    with TestRunner(args.config_dir) as runner:
        tools = runner.describe()["tools"]

    if args.json:
        print(json.dumps(tools, indent=2))
        return 0

    print("\n" + _LINE)
    print("  AVAILABLE TOOLS")
    print(_LINE + "\n")
    for tool in tools:
        # Marked because these are the ones that can change console state -
        # useful to see at a glance which agents were granted real power.
        flag = " [HARDWARE]" if tool["mutates_hardware"] else ""
        print(f"  {tool['name']}{flag}")
        print(f"    tags: {', '.join(tool['tags'])}")
        print(f"    {tool['description']}\n")
    return 0


def cmd_interactive(args: argparse.Namespace) -> int:
    """A REPL for iterating on scenarios without restarting the framework.

    Worth having because startup opens the capture card (~700ms) and re-reads
    every config; keeping one session alive makes trying five phrasings of a
    scenario quick instead of tedious.
    """
    print("\n" + _LINE)
    print("  AGENTIC CONSOLE TESTING - interactive")
    print(_LINE)
    print("\n  Type a scenario in plain English and press Enter.")
    print("  Commands:  health | info | tools | quit\n")

    with TestRunner(args.config_dir) as runner:
        while True:
            try:
                line = input("scenario> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not line:
                continue
            low = line.lower()
            if low in ("q", "quit", "exit"):
                break
            if low == "health":
                print_health(runner)
                continue
            if low == "info":
                cmd_info(args)
                continue
            if low == "tools":
                cmd_tools(args)
                continue
            print_report(runner.run(line, "text"))
    print("bye")
    return 0


# ===========================================================================
# Entry point
# ===========================================================================
def main() -> int:
    ap = argparse.ArgumentParser(
        prog="console.py",
        description="Multi-agent console testing. Describe a test in plain "
                    "English; agents check the rig, plan it, run it, verify it "
                    "against real video, and report with evidence.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python console.py health
  python console.py run "Press A and confirm the screen changes"
  python console.py run --file scenarios/dashboard-navigation.yaml
  python console.py run --requirement-file requirements/open-guide.yaml
  python console.py run "Open the guide" --dry-run
  python console.py interactive

exit codes:
  0 pass    1 fail    2 blocked (rig broken)    3 inconclusive    4 error
""")

    ap.add_argument("--config-dir", default=None,
                    help="override the config directory")

    sub = ap.add_subparsers(dest="command")

    p_run = sub.add_parser("run", help="run a scenario")
    p_run.add_argument("scenario", nargs="*", help="scenario in plain English")
    p_run.add_argument("--file", "-f", help="path to a scenario YAML file")
    p_run.add_argument("--requirement-file",
                       help="path to a minimal requirement YAML file")
    p_run.add_argument("--dry-run", action="store_true",
                       help="plan without touching hardware")
    p_run.add_argument("--max-steps", type=int, default=None)
    p_run.add_argument("--no-artifacts", action="store_true",
                       help="do not save frames or reports")

    sub.add_parser("health", help="check the rig and exit")

    p_info = sub.add_parser("info", help="show the configured setup")
    p_info.add_argument("--json", action="store_true")

    p_tools = sub.add_parser("tools", help="list the available tools")
    p_tools.add_argument("--json", action="store_true")

    sub.add_parser("interactive", help="scenario REPL")

    args = ap.parse_args()

    handlers = {
        "run": cmd_run,
        "health": cmd_health,
        "info": cmd_info,
        "tools": cmd_tools,
        "interactive": cmd_interactive,
    }
    if args.command not in handlers:
        ap.print_help()
        print("\nStart with:  python console.py health")
        return 4

    try:
        return handlers[args.command](args)
    except FileNotFoundError as exc:
        print(f"\nERROR: {exc}")
        return 4
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 4
    except Exception as exc:
        # Framework failures are exit code 4, never 1. A crash here is our bug,
        # and it must not be recorded as a console test failure.
        print(f"\nFRAMEWORK ERROR: {exc.__class__.__name__}: {exc}")
        if "--traceback" in sys.argv:
            raise
        print("Re-run with --traceback for the full stack trace.")
        return 4


if __name__ == "__main__":
    sys.exit(main())
