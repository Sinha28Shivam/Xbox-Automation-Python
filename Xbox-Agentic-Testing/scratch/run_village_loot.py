"""Run the village chest-loot + villager-trade vision loop from the current game state."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.adapters import HardwareBridge
from core.artifacts import ArtifactStore
from core.config import Config, load_dotenv_if_present
from tools.registry import ToolContext
from tools.minecraft_gameplay import village_loot_and_trade_impl

max_cycles = int(sys.argv[1]) if len(sys.argv) > 1 else 60
load_dotenv_if_present(ROOT / ".env")
settings = Config.load_all(ROOT / "config", {"settings": "settings.yaml"}, base=ROOT)["settings"]
hw = HardwareBridge(settings)
store = ArtifactStore(root=settings.resolve_path("paths.artifacts_dir", "./artifacts"),
                      run_id="village-run", frame_format="png", enabled=True)
ctx = ToolContext(hardware=hw, artifacts=store, settings=settings, dry_run=False)
result = village_loot_and_trade_impl(ctx, max_cycles=max_cycles, cycle_delay=0.4)
slim = {k: v for k, v in result.items() if k != "cycles"}
slim["ok"] = bool(result.get("goal_met"))
print("\nGOAL_MET (model-claimed, verify from frames):", result.get("goal_met"))
print("\nRESULT:", json.dumps(slim, indent=2, default=str)[:3000])
