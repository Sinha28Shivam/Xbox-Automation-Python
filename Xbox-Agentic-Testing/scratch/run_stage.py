"""Run one gameplay stage from the current game state.

usage: run_stage.py <stage> [cycles]
  explore   explore_and_map_impl        (walk the map, log sightings)
  village   village_loot_and_trade_impl (find village, chests, villagers)
  chest     loot_chest_grid             (chest already open; Y-quick-move, pixel-verified)
  attack    engage_single_mob_impl      (fight one hostile mob)
  wood      vision_guided_minecraft_gameplay_impl (tree -> logs -> planks)
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

STAGES = {"explore": 30, "village": 60, "chest": 0, "attack": 15, "wood": 45}
stage = sys.argv[1] if len(sys.argv) > 1 else ""
if stage not in STAGES:
    sys.exit(f"stage must be one of {list(STAGES)}")
cycles = int(sys.argv[2]) if len(sys.argv) > 2 else STAGES[stage]

load_dotenv_if_present(ROOT / ".env")
settings = Config.load_all(ROOT / "config", {"settings": "settings.yaml"}, base=ROOT)["settings"]
hw = HardwareBridge(settings)
store = ArtifactStore(root=settings.resolve_path("paths.artifacts_dir", "./artifacts"),
                      run_id=f"{stage}-run", frame_format="png", enabled=True)
ctx = ToolContext(hardware=hw, artifacts=store, settings=settings, dry_run=False)

if stage == "explore":
    from exploration_tools import explore_and_map_impl
    result = explore_and_map_impl(ctx, max_cycles=cycles)
elif stage == "village":
    from minecraft_gameplay import village_loot_and_trade_impl
    result = village_loot_and_trade_impl(ctx, max_cycles=cycles, cycle_delay=0.4)
elif stage == "chest":
    from chest_loot import loot_chest_grid
    result = loot_chest_grid(ctx)
elif stage == "attack":
    from combat_tools import engage_single_mob_impl
    result = engage_single_mob_impl(ctx, max_cycles=cycles)
else:
    from minecraft_gameplay import vision_guided_minecraft_gameplay_impl
    result = vision_guided_minecraft_gameplay_impl(ctx, max_cycles=cycles, cycle_delay=0.4)

slim = {k: v for k, v in result.items() if k != "cycles"}
print("\nRESULT:", json.dumps(slim, indent=2, default=str)[:3000])
