"""Self-test: verifies the config loads and names resolve. Sends nothing."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_controller import ConsolePad  # noqa: E402

pad = ConsolePad(dry_run=True)
print("LIBRARY IMPORT OK\n")

print("-- alias resolution --")
for name in ["a", "cross", "confirm", "xbox", "home", "dpad_up", "u",
             "hamburger", "windows", "left_bumper", "lt", "rsb"]:
    print(f"  {name:<14} -> {pad.cfg.gimx_name(name)}")

print("\n-- console profiles --")
for cname in pad.cfg.consoles:
    p = pad.cfg.console_profile(cname)
    print(f"  {cname:<10} type={p.get('gimx_type'):<10} fw={p.get('firmware')}")

print("\n-- unknown control handling --")
try:
    pad.cfg.gimx_name("nonexistent_button")
    print("  ERROR: should have raised!")
except KeyError as exc:
    print(f"  correctly raised KeyError: {str(exc)[:60]}...")

print("\n-- unknown console handling --")
try:
    pad.cfg.console_profile("nintendo_64")
    print("  ERROR: should have raised!")
except KeyError as exc:
    print(f"  correctly raised KeyError: {str(exc)[:60]}...")

print("\n-- macro (dry-run) --")
pad.run_macro("confirm")

print("\nALL CHECKS PASSED")
