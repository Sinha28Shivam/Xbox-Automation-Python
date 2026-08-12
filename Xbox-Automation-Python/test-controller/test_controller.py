"""
test_controller.py - Universal, config-driven console controller.

    this script -> gimx CLIENT --UDP--> gimx SERVER -> COM8 -> Leonardo -> Console

Everything (button names, stick axes, trigger ranges, timings, macros, console
profiles) lives in  config/controls.yaml  - no control names are hardcoded here.

Usable both as a CLI and as an importable library:

    from test_controller import ConsolePad
    pad = ConsolePad()
    pad.press("a")
    pad.press_times("down", 3)                  # move down 3 rows
    pad.press_times("right", 5, interval=0.4)   # slower, for animated menus
    pad.hold("guide", 2.0)
    pad.stick("left_stick", "right", duration=1.0)
    pad.trigger("rt", 255, duration=0.5)
    pad.run_macro("nav_test")

REPEATING A BUTTON (2x, 3x, ...)
    Four equivalent ways to press Down three times:
        python test_controller.py press down*3
        python test_controller.py press down 3
        python test_controller.py press down down down
        pad.press_times("down", 3)
    Mix freely:   press down*3 right*2 a
    Repeat all:   press down right --times 2
    In YAML:      - { button: down, times: 3, interval: 0.4 }

CLI:
    python test_controller.py --list
    python test_controller.py press a
    python test_controller.py press down*3 right a
    python test_controller.py hold guide 2.0
    python test_controller.py stick left_stick right --duration 1.0
    python test_controller.py trigger rt 255
    python test_controller.py macro nav_test
    python test_controller.py --interactive
    python test_controller.py --dry-run press a

PREREQUISITES
-------------
This script only SENDS input. Starting the GIMX session is a separate job,
handled by  gimx-session/gimx_session.py  - run that first, in its own terminal:

    python ../gimx-session/gimx_session.py start     # then hold GUIDE 2s

Then check from here:

    python test_controller.py --check

Without an authenticated session, events are accepted but never reach the
console.

HONEST LIMITATION
-----------------
A successful send only proves GIMX accepted the event. It does NOT prove the
console reacted. Real pass/fail needs the capture card + perception layer.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    sys.exit("PyYAML is required:  pip install pyyaml")

# --------------------------------------------------------------------------
# Config loading
# --------------------------------------------------------------------------
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "controls.yaml"

log = logging.getLogger("test_controller")


class ControlConfig:
    """Loads controls.yaml and resolves friendly names -> GIMX control names."""

    def __init__(self, path: Path | str = DEFAULT_CONFIG_PATH):
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"Control config not found: {self.path}")
        with self.path.open("r", encoding="utf-8") as fh:
            self.data: dict[str, Any] = yaml.safe_load(fh)

        self.buttons = self.data.get("buttons", {}) or {}
        self.triggers = self.data.get("triggers", {}) or {}
        self.sticks = self.data.get("sticks", {}) or {}
        self.timing = self.data.get("timing", {}) or {}
        self.macros = self.data.get("macros", {}) or {}
        self.special = self.data.get("special_actions", {}) or {}
        self.consoles = self.data.get("consoles", {}) or {}
        self.connection = self.data.get("connection", {}) or {}

        self._alias_map = self._build_alias_map()

    # -- name resolution ---------------------------------------------------
    def _build_alias_map(self) -> dict[str, tuple[str, str]]:
        """alias/name -> (kind, canonical_name) where kind is button|trigger."""
        amap: dict[str, tuple[str, str]] = {}
        for kind, table in (("button", self.buttons), ("trigger", self.triggers)):
            for canonical, spec in table.items():
                amap[canonical.lower()] = (kind, canonical)
                for alias in (spec.get("aliases") or []):
                    amap.setdefault(str(alias).lower(), (kind, canonical))
        return amap

    def resolve(self, name: str) -> tuple[str, str]:
        """Return (kind, canonical). Raises KeyError with a helpful message."""
        key = str(name).strip().lower()
        if key not in self._alias_map:
            known = ", ".join(sorted(self._alias_map))
            raise KeyError(f"Unknown control '{name}'. Known: {known}")
        return self._alias_map[key]

    def gimx_name(self, name: str) -> str:
        kind, canonical = self.resolve(name)
        table = self.buttons if kind == "button" else self.triggers
        return table[canonical]["gimx"]

    def timing_value(self, key: str, fallback: float) -> float:
        try:
            return float(self.timing.get(key, fallback))
        except (TypeError, ValueError):
            return fallback

    def console_profile(self, name: str | None = None) -> dict[str, Any]:
        """Named profile, else the one marked default, else the first."""
        if name:
            if name not in self.consoles:
                raise KeyError(
                    f"Unknown console '{name}'. Known: {', '.join(self.consoles)}")
            return self.consoles[name]
        for spec in self.consoles.values():
            if spec.get("default"):
                return spec
        if self.consoles:
            return next(iter(self.consoles.values()))
        raise KeyError("No console profiles defined in config")


# --------------------------------------------------------------------------
# The pad
# --------------------------------------------------------------------------
class ConsolePad:
    """Sends controller input to a running, authenticated GIMX session."""

    def __init__(self, config: ControlConfig | None = None,
                 console: str | None = None, dry_run: bool = False,
                 config_path: Path | str = DEFAULT_CONFIG_PATH):
        self.cfg = config or ControlConfig(config_path)
        self.profile = self.cfg.console_profile(console)
        self.dry_run = dry_run

        conn = self.cfg.connection
        self.gimx_exe = conn.get("gimx_exe", r"C:\Program Files\GIMX\gimx.exe")
        self.addr = f"{conn.get('udp_address', '127.0.0.1')}:{conn.get('udp_port', 51914)}"
        self.ctype = self.profile.get("gimx_type", "XOnePad")

        self.gap = self.cfg.timing_value("gap_between_presses", 0.25)
        self.max_retries = int(self.cfg.timing.get("udp_max_retries", 6))
        self.retry_delay = self.cfg.timing_value("udp_retry_delay", 1.5)

        self.failed = False
        self.action_log: list[dict[str, Any]] = []

    # -- low level ---------------------------------------------------------
    def _send_event(self, control: str, value: int, label: str = "") -> bool:
        """Send one raw GIMX event. Returns True if GIMX accepted it."""
        cmd = [self.gimx_exe, "--type", self.ctype,
               "--event", f"{control}({value})", "--dst", self.addr]

        if self.dry_run:
            print(f"  [dry-run] {label or control:<14} {' '.join(cmd)}")
            self._record(label or control, control, value, "dry-run")
            return True

        out = ""
        res = None
        for _ in range(self.max_retries):
            try:
                res = subprocess.run(cmd, capture_output=True, text=True,
                                     timeout=15)
            except (subprocess.TimeoutExpired, OSError) as exc:
                print(f"   FAILED: {exc}")
                self.failed = True
                self._record(label or control, control, value, f"error: {exc}")
                return False
            out = (res.stdout or "") + (res.stderr or "")
            if "can't get controller type from remote gimx" in out:
                print(".", end="", flush=True)
                time.sleep(self.retry_delay)
                continue
            break

        bad = (res is None or res.returncode != 0
               or "Error" in out or "failed" in out)
        if bad:
            print("   FAILED")
            for line in out.splitlines():
                if line.strip():
                    print(f"       {line.strip()}")
            if "Bad axis name" in out or "Bad button" in out:
                print(f"       >> '{control}' is not a valid GIMX control name.")
            self.failed = True
            self._record(label or control, control, value, "failed")
            return False

        self._record(label or control, control, value, "sent")
        return True

    def _record(self, name: str, control: str, value: int, status: str) -> None:
        self.action_log.append({
            "time": time.time(), "name": name, "gimx": control,
            "value": value, "status": status,
        })

    # -- buttons -----------------------------------------------------------
    def press(self, name: str, duration: float | None = None) -> bool:
        """Press and release a button (or fully actuate a trigger)."""
        kind, canonical = self.cfg.resolve(name)
        if duration is None:
            duration = self.cfg.timing_value("press_duration", 0.15)

        if kind == "trigger":
            spec = self.cfg.triggers[canonical]
            return self.trigger(canonical, spec.get("default_press", 255),
                                duration)

        control = self.cfg.buttons[canonical]["gimx"]
        print(f"  -> {canonical} ({control}) {duration:.2f}s", end="", flush=True)
        ok = self._send_event(control, 1, canonical)
        if not ok:
            return False
        time.sleep(duration)
        ok = self._send_event(control, 0, f"{canonical}:release")
        print("   ok" if ok else "")
        time.sleep(self.gap)
        return ok

    def press_times(self, name: str, times: int = 1,
                    duration: float | None = None,
                    interval: float | None = None) -> bool:
        """Press a button N times (e.g. move down 3 rows in a menu).

        `interval` is the pause BETWEEN repeats. It defaults to the normal
        gap; menus that animate may need a larger value or presses get eaten.
        """
        times = max(1, int(times))
        for i in range(times):
            if times > 1:
                print(f"  [{i + 1}/{times}]", end=" ")
            if not self.press(name, duration):
                return False
            if interval is not None and i < times - 1:
                time.sleep(float(interval))
        return True

    @staticmethod
    def parse_repeat(token: str) -> tuple[str, int]:
        """Parse repeat syntax into (name, count).

        Accepted:  down*3   down x3   down:3   3*down   (and plain 'down')
        """
        t = str(token).strip()
        for sep in ("*", "x", "X", ":"):
            if sep in t:
                left, _, right = t.partition(sep)
                left, right = left.strip(), right.strip()
                if right.isdigit() and left:          # down*3
                    return left, int(right)
                if left.isdigit() and right:          # 3*down
                    return right, int(left)
        return t, 1

    def hold(self, name: str, duration: float) -> bool:
        """Hold a button for an explicit duration (long / deep press)."""
        return self.press(name, duration=duration)

    def tap(self, name: str) -> bool:
        return self.press(name, self.cfg.timing_value("tap_duration", 0.08))

    def long_press(self, name: str) -> bool:
        return self.press(name, self.cfg.timing_value("long_press_duration", 1.0))

    def deep_press(self, name: str) -> bool:
        return self.press(name, self.cfg.timing_value("deep_press_duration", 2.0))

    # -- triggers ----------------------------------------------------------
    def trigger(self, name: str, value: int | None = None,
                duration: float | None = None) -> bool:
        """Analog trigger. value 0..255."""
        _, canonical = self.cfg.resolve(name)
        spec = self.cfg.triggers.get(canonical)
        if spec is None:
            print(f"  ! '{name}' is not a trigger")
            return False
        if value is None:
            value = spec.get("default_press", 255)
        value = max(spec.get("min", 0), min(spec.get("max", 255), int(value)))
        if duration is None:
            duration = self.cfg.timing_value("press_duration", 0.15)

        control = spec["gimx"]
        print(f"  -> {canonical} ({control}) = {value} for {duration:.2f}s",
              end="", flush=True)
        if not self._send_event(control, value, canonical):
            return False
        time.sleep(duration)
        ok = self._send_event(control, spec.get("min", 0), f"{canonical}:release")
        print("   ok" if ok else "")
        time.sleep(self.gap)
        return ok

    # -- sticks ------------------------------------------------------------
    def stick(self, stick_name: str, direction: str | None = None,
              x: int | None = None, y: int | None = None,
              duration: float | None = None) -> bool:
        """Move a stick either by named direction or explicit x/y (-128..127)."""
        spec = self.cfg.sticks.get(stick_name)
        if spec is None:
            known = ", ".join(self.cfg.sticks)
            print(f"  ! Unknown stick '{stick_name}'. Known: {known}")
            return False
        if duration is None:
            duration = self.cfg.timing_value("press_duration", 0.15)

        moves: list[tuple[str, int]] = []
        if direction:
            dirs = spec.get("directions", {})
            if direction not in dirs:
                print(f"  ! Unknown direction '{direction}'. "
                      f"Known: {', '.join(dirs)}")
                return False
            moves.append((dirs[direction]["axis"], int(dirs[direction]["value"])))
        else:
            lo, hi = spec.get("min", -128), spec.get("max", 127)
            if x is not None:
                moves.append((spec["x_axis"], max(lo, min(hi, int(x)))))
            if y is not None:
                moves.append((spec["y_axis"], max(lo, min(hi, int(y)))))
        if not moves:
            print("  ! stick() needs a direction or x/y")
            return False

        label = direction or f"x={x},y={y}"
        print(f"  -> {stick_name} {label} for {duration:.2f}s", end="", flush=True)
        for axis, val in moves:
            if not self._send_event(axis, val, f"{stick_name}:{label}"):
                return False
        time.sleep(duration)
        center = spec.get("center", 0)
        ok = True
        for axis, _ in moves:
            ok = self._send_event(axis, center, f"{stick_name}:center") and ok
        print("   ok" if ok else "")
        time.sleep(self.gap)
        return ok

    # -- sequences ---------------------------------------------------------
    def sequence(self, steps: list[dict[str, Any]]) -> bool:
        """Run a list of steps: {button|trigger|stick|wait, ...}."""
        for step in steps:
            if "wait" in step:
                time.sleep(float(step["wait"]))
                continue
            if "button" in step:
                if not self.press_times(step["button"],
                                        step.get("times", step.get("repeat", 1)),
                                        step.get("duration"),
                                        step.get("interval")):
                    return False
            elif "trigger" in step:
                if not self.trigger(step["trigger"], step.get("value"),
                                    step.get("duration")):
                    return False
            elif "stick" in step:
                if not self.stick(step["stick"], step.get("direction"),
                                  step.get("x"), step.get("y"),
                                  step.get("duration")):
                    return False
            else:
                print(f"  ! unrecognized step: {step}")
                return False
        return True

    def run_macro(self, macro_name: str) -> bool:
        macro = self.cfg.macros.get(macro_name)
        if macro is None:
            print(f"  ! Unknown macro '{macro_name}'. "
                  f"Known: {', '.join(self.cfg.macros)}")
            return False
        print(f"\n=== macro: {macro_name} - {macro.get('description', '')} ===")
        return self.sequence(macro.get("steps", []))

    def run_special(self, action_name: str) -> bool:
        action = self.cfg.special.get(action_name)
        if action is None:
            print(f"  ! Unknown action '{action_name}'. "
                  f"Known: {', '.join(self.cfg.special)}")
            return False
        if not action.get("verified", False):
            print(f"  NOTE: '{action_name}' is NOT hardware-verified - "
                  f"the sequence is a best guess.")
        print(f"\n=== action: {action_name} - {action.get('description','')} ===")
        return self.sequence(action.get("sequence", []))


# --------------------------------------------------------------------------
# Session check
# --------------------------------------------------------------------------
# Session management lives in gimx-session/gimx_session.py. We reuse it here
# rather than duplicating the logic, so there is one definition of "is the
# session up?". The import is optional so this file still works standalone.
_SESSION_DIR = Path(__file__).resolve().parent.parent / "gimx-session"
if str(_SESSION_DIR) not in sys.path:
    sys.path.insert(0, str(_SESSION_DIR))
try:
    from gimx_session import session_is_up as _session_is_up
except ImportError:          # pragma: no cover - fallback if moved/missing
    _session_is_up = None


def check_session(pad: ConsolePad) -> bool:
    """Verify a GIMX server is reachable on the configured UDP address."""
    print(f"Checking for a GIMX session at {pad.addr} ...")

    if _session_is_up is not None:
        ok = _session_is_up(pad.cfg.path, quiet=False)
    else:
        # Fallback: ask GIMX directly with a harmless no-op event.
        cmd = [pad.gimx_exe, "--type", pad.ctype, "--event", "up(0)",
               "--dst", pad.addr]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True,
                                 timeout=15)
        except (subprocess.TimeoutExpired, OSError) as exc:
            print(f"  cannot run gimx.exe: {exc}")
            return False
        out = (res.stdout or "") + (res.stderr or "")
        ok = "Remote GIMX detected" in out
        if not ok:
            for line in out.splitlines():
                if line.strip():
                    print(f"    {line.strip()}")

    if ok:
        print("  Reminder: reachable is NOT the same as authenticated. If the")
        print("  Guide button was never held for 2s, events report 'ok' but")
        print("  never reach the console.")
    else:
        print("  Start one with:")
        print("      python ../gimx-session/gimx_session.py start")
    return ok


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def print_controls(cfg: ControlConfig) -> None:
    print("=" * 62)
    print(f"CONTROL MAP   ({cfg.path})")
    print("=" * 62)
    print("\nBUTTONS            GIMX      ALIASES")
    for name, spec in cfg.buttons.items():
        print(f"  {name:<16} {spec['gimx']:<9} {', '.join(spec.get('aliases', []))}")
    print("\nTRIGGERS (analog)  GIMX      RANGE")
    for name, spec in cfg.triggers.items():
        print(f"  {name:<16} {spec['gimx']:<9} "
              f"{spec.get('min',0)}..{spec.get('max',255)}")
    print("\nSTICKS             AXES                  DIRECTIONS")
    for name, spec in cfg.sticks.items():
        print(f"  {name:<16} {spec['x_axis']} / {spec['y_axis']:<10} "
              f"{', '.join(spec.get('directions', {}))}")
    print("\nMACROS")
    for name, spec in cfg.macros.items():
        print(f"  {name:<16} {spec.get('description','')}")
    print("\nSPECIAL ACTIONS")
    for name, spec in cfg.special.items():
        mark = "" if spec.get("verified") else "  [UNVERIFIED]"
        print(f"  {name:<16} {spec.get('description','')}{mark}")
    print("\nCONSOLES")
    for name, spec in cfg.consoles.items():
        mark = "  (default)" if spec.get("default") else ""
        print(f"  {name:<16} type={spec.get('gimx_type')} "
              f"firmware={spec.get('firmware')}{mark}")
    print("\nTIMING (seconds)")
    for k, v in cfg.timing.items():
        print(f"  {k:<26} {v}")


def interactive(pad: ConsolePad) -> None:
    print("\nInteractive mode. Examples:")
    print("  a                 press A")
    print("  down*3            press Down 3 times")
    print("  down 3            same thing")
    print("  down*2 right a    combine in one line")
    print("  hold guide 2      hold Guide 2s")
    print("  stick left right  move left stick right")
    print("  trigger rt 255    pull right trigger")
    print("  macro nav_test    run a macro")
    print("  q                 quit\n")
    while True:
        try:
            line = input("pad> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue
        if line.lower() in ("q", "quit", "exit"):
            break
        parts = line.split()
        verb = parts[0].lower()
        try:
            if verb == "hold" and len(parts) >= 3:
                pad.hold(parts[1], float(parts[2]))
            elif verb == "stick" and len(parts) >= 3:
                name = parts[1] if parts[1] in pad.cfg.sticks else f"{parts[1]}_stick"
                pad.stick(name, parts[2])
            elif verb == "trigger" and len(parts) >= 2:
                pad.trigger(parts[1], int(parts[2]) if len(parts) > 2 else None)
            elif verb == "macro" and len(parts) >= 2:
                pad.run_macro(parts[1])
            elif verb == "action" and len(parts) >= 2:
                pad.run_special(parts[1])
            else:
                idx = 0
                while idx < len(parts):
                    nm, count = ConsolePad.parse_repeat(parts[idx])
                    # allow "down 3" with the count as a separate token
                    if (count == 1 and idx + 1 < len(parts)
                            and parts[idx + 1].isdigit()):
                        count = int(parts[idx + 1])
                        idx += 1
                    pad.press_times(nm, count)
                    idx += 1
        except KeyError as exc:
            print(f"  ! {exc}")
        except ValueError as exc:
            print(f"  ! bad number: {exc}")
    print("bye")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Universal config-driven console controller (via GIMX).",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG_PATH),
                    help="path to controls.yaml")
    ap.add_argument("--console", default=None,
                    help="console profile (default: the one marked default)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print commands without sending")
    ap.add_argument("--list", action="store_true", help="show the control map")
    ap.add_argument("--check", action="store_true",
                    help="check that a GIMX session is reachable")
    ap.add_argument("--interactive", action="store_true")

    sub = ap.add_subparsers(dest="command")

    p_press = sub.add_parser(
        "press", help="press buttons; supports repeats like  down*3  or  down 3")
    p_press.add_argument("names", nargs="+",
                         help="e.g. down*3 right a   (or:  down 3 right a)")
    p_press.add_argument("--times", type=int, default=None,
                         help="repeat EVERY listed button N times")
    p_press.add_argument("--interval", type=float, default=None,
                         help="extra pause between repeats (seconds)")
    p_press.add_argument("--duration", type=float, default=None)

    p_hold = sub.add_parser("hold", help="hold a button for N seconds")
    p_hold.add_argument("name")
    p_hold.add_argument("duration", type=float)

    p_stick = sub.add_parser("stick", help="move a stick")
    p_stick.add_argument("stick_name")
    p_stick.add_argument("direction", nargs="?", default=None)
    p_stick.add_argument("--x", type=int, default=None)
    p_stick.add_argument("--y", type=int, default=None)
    p_stick.add_argument("--duration", type=float, default=None)

    p_trig = sub.add_parser("trigger", help="pull an analog trigger")
    p_trig.add_argument("name")
    p_trig.add_argument("value", type=int, nargs="?", default=None)
    p_trig.add_argument("--duration", type=float, default=None)

    p_macro = sub.add_parser("macro", help="run a macro from the config")
    p_macro.add_argument("name")

    p_action = sub.add_parser("action", help="run a special action")
    p_action.add_argument("name")

    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    try:
        cfg = ControlConfig(args.config)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        return 2

    if args.list:
        print_controls(cfg)
        return 0

    try:
        pad = ConsolePad(cfg, args.console, args.dry_run)
    except KeyError as exc:
        print(f"ERROR: {exc}")
        return 2

    if args.check:
        return 0 if check_session(pad) else 1

    if not args.dry_run:
        print(f"console={args.console or 'default'} type={pad.ctype} "
              f"-> GIMX at {pad.addr}")

    if args.interactive:
        interactive(pad)
        return 1 if pad.failed else 0

    try:
        if args.command == "press":
            # Accepts "down*3 a" and "down 3 a": a bare integer applies to
            # the button token before it.
            tokens: list[list[Any]] = []
            for raw in args.names:
                if raw.isdigit() and tokens:
                    tokens[-1][1] = int(raw)
                    continue
                nm, count = ConsolePad.parse_repeat(raw)
                tokens.append([nm, count])

            for nm, count in tokens:
                if args.times is not None:
                    count = args.times
                if not pad.press_times(nm, count, args.duration, args.interval):
                    break
        elif args.command == "hold":
            pad.hold(args.name, args.duration)
        elif args.command == "stick":
            pad.stick(args.stick_name, args.direction, args.x, args.y,
                      args.duration)
        elif args.command == "trigger":
            pad.trigger(args.name, args.value, args.duration)
        elif args.command == "macro":
            pad.run_macro(args.name)
        elif args.command == "action":
            pad.run_special(args.name)
        else:
            ap.print_help()
            print("\nTip: start with  --list  then  --check")
            return 1
    except KeyError as exc:
        print(f"ERROR: {exc}")
        return 2

    return 1 if pad.failed else 0


if __name__ == "__main__":
    sys.exit(main())
