"""Run the Minecraft vision gameplay loop directly from the current game state."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for sub in ("core", "tools", "agents", "graph"):
    sys.path.insert(0, str(ROOT / sub))

from adapters import HardwareBridge
from artifacts import ArtifactStore
from config import Config, load_dotenv_if_present
from registry import ToolContext
from minecraft_gameplay import vision_guided_minecraft_gameplay_impl

max_cycles = int(sys.argv[1]) if len(sys.argv) > 1 else 120
load_dotenv_if_present(ROOT / ".env")
settings = Config.load_all(ROOT / "config", {"settings": "settings.yaml"}, base=ROOT)["settings"]
hw = HardwareBridge(settings)
store = ArtifactStore(root=settings.resolve_path("paths.artifacts_dir", "./artifacts"),
                      run_id="explore-run", frame_format="png", enabled=True)
ctx = ToolContext(hardware=hw, artifacts=store, settings=settings, dry_run=False)
result = vision_guided_minecraft_gameplay_impl(ctx, max_cycles=max_cycles, cycle_delay=0.4)
slim = {k: v for k, v in result.items() if k != "cycles"}
print("\nRESULT:", json.dumps(slim, indent=2, default=str)[:3000])
