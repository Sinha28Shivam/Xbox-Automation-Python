"""Fast Minecraft Dungeons loop, optionally driving several controllers.

usage: run_dungeons_fast.py [cycles] [--udp-ports 51914,51915]

Speed: small JPEG to the LLM, short max_tokens, pipelined frames (one grab
per cycle, background disk writes), no inter-cycle delay.

Multi-controller: each extra UDP port must be a separately running,
authenticated GIMX server (its own Leonardo / COM port) whose player has
joined co-op. Inputs are MIRRORED to every pad in parallel - there is one
capture card, so all heroes act on the same decision.
"""
import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for sub in ("core", "tools", "agents", "graph", "tools/game_profiles"):
    sys.path.insert(0, str(ROOT / sub))

os.environ.setdefault("ANTHROPIC_MAX_TOKENS", "1024")

from adapters import HardwareBridge
from artifacts import ArtifactStore
from config import Config, load_dotenv_if_present
from registry import ToolContext
import gameplay_engine
from gameplay_engine import run_gameplay_loop
from game_profiles.dungeons_agents import PROFILE, ReporterAgent

gameplay_engine.VISION_WIDTH = 768
gameplay_engine.JPEG_QUALITY = 70


class MirrorPad:
    """Looks like one ConsolePad; every method call fans out to all pads."""

    def __init__(self, pads):
        self.pads = pads
        self._pool = ThreadPoolExecutor(max_workers=len(pads))

    def __getattr__(self, name):
        attr = getattr(self.pads[0], name)
        if not callable(attr):
            return attr  # cfg, addr, ... come from the primary pad

        def fan_out(*args, **kwargs):
            futures = [self._pool.submit(getattr(p, name), *args, **kwargs)
                       for p in self.pads]
            results = [f.result() for f in futures]
            return results[0]
        return fan_out


ap = argparse.ArgumentParser()
ap.add_argument("cycles", nargs="?", type=int, default=40)
ap.add_argument("--udp-ports", default="",
                help="comma-separated GIMX UDP ports, one per controller")
args = ap.parse_args()

load_dotenv_if_present(ROOT / ".env")
settings = Config.load_all(ROOT / "config", {"settings": "settings.yaml"}, base=ROOT)["settings"]
hw = HardwareBridge(settings)

ports = [int(p) for p in args.udp_ports.split(",") if p.strip()]
if len(ports) > 1:
    primary = hw.pad()
    host = primary.addr.split(":")[0]
    pads = []
    for port in ports:
        p = type(primary)(hw.controls, None, hw.dry_run)
        p.addr = f"{host}:{port}"
        pads.append(p)
    hw._pad = MirrorPad(pads)
    print(f"[pads] mirroring to {[p.addr for p in pads]}", flush=True)

store = ArtifactStore(root=settings.resolve_path("paths.artifacts_dir", "./artifacts"),
                      run_id="dungeons-fast", frame_format="jpg", enabled=True)
ctx = ToolContext(hardware=hw, artifacts=store, settings=settings, dry_run=False)

result = run_gameplay_loop(ctx, PROFILE, goal=PROFILE.default_goal,
                           max_cycles=args.cycles, cycle_delay=0.0,
                           settle_after_move=0.1, pipeline=True,
                           overlap_act=True)
slim = {k: v for k, v in result.items() if k != "cycles"}
print("\nRESULT:", json.dumps(slim, indent=2, default=str)[:3000])
ReporterAgent().run(result, store.reports_dir)
