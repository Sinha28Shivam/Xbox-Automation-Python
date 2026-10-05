"""Run the Minecraft Dungeons vision loop from the current game state.

usage: run_dungeons.py [cycles]   (game must already be on a mission)
Bindings: config/dungeons_controls.yaml (from the in-game Controls screen).
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for sub in ("core", "tools", "agents", "graph", "tools/game_profiles"):
    sys.path.insert(0, str(ROOT / sub))

from adapters import HardwareBridge
from artifacts import ArtifactStore
from config import Config, load_dotenv_if_present
from registry import ToolContext
import gameplay_engine
from gameplay_engine import run_gameplay_loop
from game_profiles.dungeons_profile import PROFILE

gameplay_engine.VISION_WIDTH = 960  # smaller image -> faster LLM round-trip

cycles = int(sys.argv[1]) if len(sys.argv) > 1 else 40

load_dotenv_if_present(ROOT / ".env")
settings = Config.load_all(ROOT / "config", {"settings": "settings.yaml"}, base=ROOT)["settings"]
hw = HardwareBridge(settings)
store = ArtifactStore(root=settings.resolve_path("paths.artifacts_dir", "./artifacts"),
                      run_id="dungeons-run", frame_format="png", enabled=True)
ctx = ToolContext(hardware=hw, artifacts=store, settings=settings, dry_run=False)

result = run_gameplay_loop(ctx, PROFILE, goal=PROFILE.default_goal,
                           max_cycles=cycles, cycle_delay=0.0,
                           settle_after_move=0.2)
slim = {k: v for k, v in result.items() if k != "cycles"}
print("\nRESULT:", json.dumps(slim, indent=2, default=str)[:3000])
