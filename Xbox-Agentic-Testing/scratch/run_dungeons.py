"""Run the Minecraft Dungeons single-decision vision loop - thin wrapper.

usage: run_dungeons.py [cycles]   (game must already be on a mission)
Prefer: python console.py dungeons --single --slow --cycles N
Bindings: config/dungeons_controls.yaml (from the in-game Controls screen).
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.adapters import HardwareBridge
from core.artifacts import ArtifactStore
from core.config import Config, load_dotenv_if_present
from tools.dungeons_gameplay import vision_guided_dungeons_gameplay_impl
from tools.registry import ToolContext

cycles = int(sys.argv[1]) if len(sys.argv) > 1 else 40

load_dotenv_if_present(ROOT / ".env")
settings = Config.load_all(ROOT / "config", {"settings": "settings.yaml"}, base=ROOT)["settings"]
store = ArtifactStore(root=settings.resolve_path("paths.artifacts_dir", "./artifacts"),
                      run_id="dungeons-run", frame_format="png", enabled=True)
ctx = ToolContext(hardware=HardwareBridge(settings), artifacts=store,
                  settings=settings, dry_run=False)

result = vision_guided_dungeons_gameplay_impl(ctx, max_cycles=cycles, agents=False, fast=False)
slim = {k: v for k, v in result.items() if k != "cycles"}
print("\nRESULT:", json.dumps(slim, indent=2, default=str)[:3000])
