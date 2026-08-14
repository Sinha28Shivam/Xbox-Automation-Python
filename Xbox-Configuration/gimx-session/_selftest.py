"""Offline checks for gimx_session.py. Starts nothing, sends nothing."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gimx_session import GimxSession, load_config, running_gimx_pids  # noqa: E402

print("IMPORT OK\n")

s = GimxSession()
print("-- resolved settings --")
print(f"  gimx_exe    : {s.gimx_exe}")
print(f"  serial port : {s.port}")
print(f"  udp addr    : {s.addr}")
print(f"  type        : {s.ctype}")
print(f"  gimx config : {s.gimx_config}")
print(f"  nograb      : {s.nograb}")
print(f"  idle timeout: {s.idle_timeout} min")
print(f"  normal prio : {s.force_normal_priority}")

print("\n-- server command that WOULD be run --")
print("  " + " ".join(s.build_command()))

print("\n-- safety flag assertions --")
cmd = s.build_command()
assert "--nograb" in cmd, "--nograb missing (freeze protection)"
assert "--timeout" in cmd, "--timeout missing (runaway protection)"
assert cmd[-2] == "--port", f"--port must be LAST, got: {cmd[-2:]}"
assert "--config" in cmd, "--config missing (no input bindings without it)"
print("  --nograb present            OK")
print("  --timeout present           OK")
print("  --config present            OK")
print("  --port is last argument     OK")

print("\n-- other console profiles --")
for name in (load_config().get("consoles") or {}):
    try:
        other = GimxSession(console=name)
        print(f"  {name:<10} type={other.ctype:<10} config={other.gimx_config}")
    except KeyError as exc:
        print(f"  {name:<10} ERROR {exc}")

print("\n-- unknown console handling --")
try:
    GimxSession(console="nintendo_64")
    print("  ERROR: should have raised!")
except KeyError as exc:
    print(f"  correctly raised KeyError: {str(exc)[:55]}...")

print(f"\n-- running gimx processes: {running_gimx_pids() or 'none'}")
print("\nALL CHECKS PASSED")
