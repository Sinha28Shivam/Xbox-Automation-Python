"""Fast Minecraft Dungeons loop - thin wrapper over the main tool.

usage: run_dungeons_fast.py [cycles] [--udp-ports 51914,51915]
Prefer: python console.py dungeons --cycles N [--udp-ports ...]
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("ANTHROPIC_MAX_TOKENS", "1024")

from core.adapters import HardwareBridge
from core.artifacts import ArtifactStore
from core.config import Config, load_dotenv_if_present
from tools.dungeons_gameplay import vision_guided_dungeons_gameplay_impl
from tools.registry import ToolContext

ap = argparse.ArgumentParser()
ap.add_argument("cycles", nargs="?", type=int, default=40)
ap.add_argument("--udp-ports", default="",
                help="comma-separated GIMX UDP ports, one per controller")
args = ap.parse_args()

load_dotenv_if_present(ROOT / ".env")
settings = Config.load_all(ROOT / "config", {"settings": "settings.yaml"}, base=ROOT)["settings"]
store = ArtifactStore(root=settings.resolve_path("paths.artifacts_dir", "./artifacts"),
                      run_id="dungeons-fast", frame_format="jpg", enabled=True)
ctx = ToolContext(hardware=HardwareBridge(settings), artifacts=store,
                  settings=settings, dry_run=False)

result = vision_guided_dungeons_gameplay_impl(ctx, max_cycles=args.cycles, agents=True,
                                              fast=True, udp_ports=args.udp_ports)
slim = {k: v for k, v in result.items() if k != "cycles"}
print("\nRESULT:", json.dumps(slim, indent=2, default=str)[:3000])
