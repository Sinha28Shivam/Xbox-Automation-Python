"""dungeons_gameplay.py - tool registration for Minecraft Dungeons vision gameplay.

The game-specific logic lives in game_profiles/dungeons_profile.py (single
vision decision per cycle) and game_profiles/dungeons_agents.py (the same
profile driven by an observer + health/combat/interaction/navigator agents).
The shared observe->decide->act->re-observe loop lives in gameplay_engine.py.
This module wires them together as a registered tool, and owns the run
options that used to live only in scratch/run_dungeons*.py:

  * fast mode       - smaller JPEG to the model, pipelined frames, overlapped
                      acting (what scratch/run_dungeons_fast.py did)
  * multi-agent     - the supervisor-driven profile + scenario report
  * extra pads      - mirror every input to several authenticated GIMX
                      servers for co-op (one capture card, one decision)

The game must already be on a mission: this tool plays, it does not launch.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import Any, Iterator

from tools import gameplay_engine
from tools.gameplay_engine import GameProfile, run_gameplay_loop
from tools.registry import ToolContext, ToolSpec, fail, make_tool

DEFAULT_GOAL = ("Follow the objective marker through the mission, fight "
                "enemies that block the way, open chests, and reach the "
                "mission-complete screen.")

# Fast mode trades image detail for round-trip latency; Dungeons' isometric
# view stays readable at this size (measured in scratch/run_dungeons_fast.py).
_FAST_VISION_WIDTH = 768
_FAST_JPEG_QUALITY = 70


def resolve_profile(agents: bool = True) -> GameProfile:
    """The multi-agent profile by default; the single-decision one otherwise."""
    if agents:
        from tools.game_profiles.dungeons_agents import PROFILE
    else:
        from tools.game_profiles.dungeons_profile import PROFILE
    return PROFILE


@contextmanager
def _encoder(fast: bool) -> Iterator[None]:
    """Shrink the frames sent to the model for this run only.

    VISION_WIDTH / JPEG_QUALITY are module globals shared by every game, so
    they are restored afterwards - a fast Dungeons run must not leave Max or
    Minecraft sending degraded frames.
    """
    if not fast:
        yield
        return
    saved = (gameplay_engine.VISION_WIDTH, gameplay_engine.JPEG_QUALITY)
    gameplay_engine.VISION_WIDTH = _FAST_VISION_WIDTH
    gameplay_engine.JPEG_QUALITY = _FAST_JPEG_QUALITY
    try:
        yield
    finally:
        gameplay_engine.VISION_WIDTH, gameplay_engine.JPEG_QUALITY = saved


class MirrorPad:
    """Looks like one ConsolePad; every method call fans out to all pads.

    Attributes (cfg, addr, ...) come from the primary pad; method calls run on
    every pad in parallel and return the primary's result.
    """

    def __init__(self, pads: list[Any]):
        self.pads = pads
        self._pool = ThreadPoolExecutor(max_workers=len(pads))

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self.pads[0], name)
        if not callable(attr):
            return attr

        def fan_out(*args: Any, **kwargs: Any) -> Any:
            futures = [self._pool.submit(getattr(p, name), *args, **kwargs)
                       for p in self.pads]
            return [f.result() for f in futures][0]
        return fan_out

    def close(self) -> None:
        self._pool.shutdown(wait=False)


def parse_ports(udp_ports: str | list[int] | None) -> list[int]:
    if not udp_ports:
        return []
    if isinstance(udp_ports, str):
        return [int(p) for p in udp_ports.replace(" ", "").split(",") if p]
    return [int(p) for p in udp_ports]


@contextmanager
def _mirrored_pads(ctx: ToolContext, ports: list[int]) -> Iterator[list[str]]:
    """Temporarily route the bridge's pad to a MirrorPad over `ports`."""
    if len(ports) < 2:
        yield []
        return
    hw = ctx.hardware
    primary = hw.pad()
    host = primary.addr.split(":")[0]
    pads = []
    for port in ports:
        pad = type(primary)(hw.controls, None, hw.dry_run)
        pad.addr = f"{host}:{port}"
        pads.append(pad)
    mirror = MirrorPad(pads)
    saved = hw._pad
    hw._pad = mirror
    try:
        yield [p.addr for p in pads]
    finally:
        hw._pad = saved
        mirror.close()


def vision_guided_dungeons_gameplay_impl(
    ctx: ToolContext,
    goal: str = DEFAULT_GOAL,
    max_cycles: int = 40,
    agents: bool = True,
    fast: bool = True,
    udp_ports: str | list[int] | None = None,
    write_report: bool = True,
) -> dict[str, Any]:
    """Play Minecraft Dungeons closed-loop until mission complete or the budget ends."""
    if ctx.dry_run:
        return fail("Dry run: Dungeons gameplay drives live hardware and was "
                    "not started.", dispatched=False, dry_run=True)

    try:
        ports = parse_ports(udp_ports)
    except ValueError:
        return fail(f"udp_ports must be comma-separated integers, got {udp_ports!r}",
                    dispatched=False)

    profile = resolve_profile(agents)
    tuning: dict[str, Any] = (
        {"cycle_delay": 0.0, "settle_after_move": 0.1, "pipeline": True,
         "overlap_act": True}
        if fast else {"cycle_delay": 0.0, "settle_after_move": 0.2})

    with _mirrored_pads(ctx, ports) as mirrored, _encoder(fast):
        if mirrored:
            print(f"[pads] mirroring to {mirrored}", flush=True)
        result = run_gameplay_loop(ctx, profile, goal=goal or DEFAULT_GOAL,
                                   max_cycles=max_cycles, **tuning)

    result["mode"] = {"agents": agents, "fast": fast, "pads": mirrored or None}

    # The scenario report reads the per-agent fields only the multi-agent
    # profile records, so it is skipped for the single-decision profile.
    if result.get("ok") and agents and write_report and ctx.artifacts.enabled:
        try:
            from tools.game_profiles.dungeons_agents import ReporterAgent
            report = ReporterAgent().run(result, ctx.artifacts.reports_dir)
            result["scenario_verdict"] = report.get("verdict")
            result["scenario_report"] = str(
                ctx.artifacts.reports_dir / "dungeons-scenarios.html")
        except Exception as exc:  # the gameplay evidence is still valid
            result["scenario_report_error"] = f"{type(exc).__name__}: {exc}"
    return result


def _vision_guided_dungeons_gameplay(ctx: ToolContext) -> Any:
    def run(goal: str = DEFAULT_GOAL, max_cycles: int = 40,
            agents: bool = True, fast: bool = True) -> dict[str, Any]:
        return vision_guided_dungeons_gameplay_impl(
            ctx, goal=goal, max_cycles=max_cycles, agents=agents, fast=fast)

    return make_tool(
        run, "vision_guided_dungeons_gameplay",
        "Play Minecraft Dungeons closed-loop with a VISION model. The game "
        "must already be on a mission. Every cycle it reads the live frame "
        "(enemies, health, loot, objective marker, minimap) and an observer "
        "hands the situation to a health, combat, interaction or navigator "
        "agent that presses the buttons. Keeps playing until the mission-"
        "complete screen is seen, max_cycles is reached, or nothing on screen "
        "changes any more. Returns per-cycle frames, deltas and reasoning as "
        "evidence, plus a scenario report (rescue, fight, loot, navigation).")


def provide() -> list[ToolSpec]:
    return [
        ToolSpec(name="vision_guided_dungeons_gameplay",
                 description=("Closed-loop vision gameplay for Minecraft Dungeons: "
                              "follow the objective, fight, loot, and reach "
                              "mission complete, with a scenario report."),
                 tags=["input", "vision", "game"],
                 factory=_vision_guided_dungeons_gameplay, mutates_hardware=True),
    ]
