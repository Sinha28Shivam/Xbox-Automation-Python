"""
gimx_session.py - Start, check and stop the GIMX session.

This is the ONE place that owns the GIMX server process, the serial port, and
the Guide-button authentication step. Everything else (test_controller.py, the
future agentic framework) just sends events to a session started here.

WHY THIS IS A SEPARATE FILE
---------------------------
Starting a session is a different kind of job from sending button presses:

  * It holds COM8 for as long as it runs (only ONE process can).
  * It needs a human to hold the controller's GUIDE button for 2 seconds.
  * It is long-lived, whereas sending a press is instant and stateless.

Mixing those two concerns made the earlier code confusing, so the session lives
here and can be run on its own in a terminal you leave open.

TWO WAYS TO USE IT
------------------
1. Standalone (recommended - run in its own terminal and leave it):

       python gimx_session.py start        # starts and streams the log
       python gimx_session.py status       # is a session up and reachable?
       python gimx_session.py stop         # stop any running GIMX
       python gimx_session.py restart

2. Imported by other code:

       from gimx_session import GimxSession, session_is_up

       if not session_is_up():
           print("Start a session first: python gimx_session.py start")

       # or manage one automatically (stops on exit):
       with GimxSession() as s:
           ...                             # send events here

SAFETY - PLEASE READ
--------------------
*** DO NOT run GIMX as administrator unless you must. ***
Elevated, gimx.exe claims REALTIME cpu priority and grabs your input devices.
During development this froze an entire PC (stuck mouse, nothing clickable).

Sessions started by this module always pass:
    --nograb                 never capture mouse/keyboard
    --timeout <minutes>      auto-exit when idle
and the process priority is forced to Normal immediately after launch.

These protections apply ONLY to sessions started by this module. A session you
launch by hand as administrator is not protected.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    sys.exit("PyYAML is required:  pip install pyyaml")

DEFAULT_CONFIG_PATH = (Path(__file__).resolve().parent.parent
                       / "config" / "controls.yaml")

# Lines GIMX prints that we care about.
RE_ADAPTER = re.compile(r"adapter detected", re.I)
RE_GUIDE = re.compile(r"guide button", re.I)
RE_FIRMWARE = re.compile(r"Firmware version:\s*(\S+)", re.I)
RE_BAUD = re.compile(r"Using baudrate:\s*(\d+)", re.I)
RE_ERROR = re.compile(r"^\s*Error:|failed", re.I)
RE_ELEVATION = re.compile(r"priority class can't be used", re.I)


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Control config not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _console_profile(cfg: dict[str, Any], name: str | None) -> dict[str, Any]:
    consoles = cfg.get("consoles", {}) or {}
    if name:
        if name not in consoles:
            raise KeyError(f"Unknown console '{name}'. "
                           f"Known: {', '.join(consoles)}")
        return consoles[name]
    for spec in consoles.values():
        if spec.get("default"):
            return spec
    if consoles:
        return next(iter(consoles.values()))
    raise KeyError("No console profiles defined in config")


# --------------------------------------------------------------------------
# Helpers usable without starting anything
# --------------------------------------------------------------------------
def session_is_up(config_path: Path | str = DEFAULT_CONFIG_PATH,
                  console: str | None = None, quiet: bool = True) -> bool:
    """True if a GIMX server is running and reachable over UDP.

    NOTE: this proves the server is *reachable*. It does NOT prove the session
    was authenticated with the Guide button, and it does NOT prove input
    actually reaches the console. Only watching the TV proves that.
    """
    cfg = load_config(config_path)
    conn = cfg.get("connection", {}) or {}
    profile = _console_profile(cfg, console)

    gimx = conn.get("gimx_exe", r"C:\Program Files\GIMX\gimx.exe")
    addr = (f"{conn.get('udp_address', '127.0.0.1')}:"
            f"{conn.get('udp_port', 51914)}")
    ctype = profile.get("gimx_type", "XOnePad")

    # "up(0)" is a no-op release of the D-pad: harmless, but it makes GIMX do
    # the full remote handshake so we learn whether the server answers.
    cmd = [gimx, "--type", ctype, "--event", "up(0)", "--dst", addr]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    except (subprocess.TimeoutExpired, OSError) as exc:
        if not quiet:
            print(f"  cannot run gimx.exe: {exc}")
        return False

    out = (res.stdout or "") + (res.stderr or "")
    ok = "Remote GIMX detected" in out
    if not quiet:
        if ok:
            print(f"  OK - GIMX session reachable at {addr}")
        else:
            print(f"  NO SESSION at {addr}")
            for line in out.splitlines():
                if line.strip():
                    print(f"    {line.strip()}")
    return ok


def running_gimx_pids() -> list[int]:
    """PIDs of any running gimx.exe (not the launcher)."""
    try:
        res = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-Process gimx -ErrorAction SilentlyContinue).Id"],
            capture_output=True, text=True, timeout=15)
    except (subprocess.TimeoutExpired, OSError):
        return []
    return [int(t) for t in (res.stdout or "").split() if t.strip().isdigit()]


def stop_all(quiet: bool = False) -> bool:
    """Stop every running gimx.exe, releasing the serial port."""
    pids = running_gimx_pids()
    if not pids:
        if not quiet:
            print("No gimx.exe running - nothing to stop.")
        return True
    if not quiet:
        print(f"Stopping gimx.exe (pid {', '.join(map(str, pids))}) ...")
    subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "Stop-Process -Name gimx -Force -ErrorAction SilentlyContinue"],
        capture_output=True, text=True)
    time.sleep(2)
    left = running_gimx_pids()
    if left:
        if not quiet:
            print(f"  Could NOT stop pid {left}. It is probably running "
                  f"elevated - close it from its own window or an admin "
                  f"Task Manager.")
        return False
    if not quiet:
        print("  Stopped. Serial port released.")
    return True


# --------------------------------------------------------------------------
# The session
# --------------------------------------------------------------------------
class GimxSession:
    """Owns a gimx.exe server process (holds the serial port, listens on UDP)."""

    def __init__(self, config_path: Path | str = DEFAULT_CONFIG_PATH,
                 console: str | None = None, port: str | None = None,
                 gimx_config: str | None = None):
        self.cfg = load_config(config_path)
        self.profile = _console_profile(self.cfg, console)
        conn = self.cfg.get("connection", {}) or {}
        timing = self.cfg.get("timing", {}) or {}
        safety = conn.get("safety", {}) or {}

        self.gimx_exe = conn.get("gimx_exe", r"C:\Program Files\GIMX\gimx.exe")
        self.port = port or conn.get("serial_port", "COM8")
        self.udp_address = conn.get("udp_address", "127.0.0.1")
        self.udp_port = int(conn.get("udp_port", 51914))
        self.addr = f"{self.udp_address}:{self.udp_port}"

        self.ctype = self.profile.get("gimx_type", "XOnePad")
        # --config is REQUIRED. Without it GIMX starts, reports the adapter,
        # and silently forwards nothing - not even the physical controller.
        self.gimx_config = gimx_config or self.profile.get("config_file")

        self.nograb = bool(safety.get("nograb", True))
        self.idle_timeout = safety.get("idle_timeout_minutes", 5)
        self.force_normal_priority = bool(
            safety.get("force_normal_priority", True))
        self.settle = float(timing.get("session_settle", 4.0))

        self.proc: subprocess.Popen[str] | None = None
        self.adapter_detected = False
        self.guide_prompted = False
        self.firmware: str | None = None
        self.baudrate: str | None = None
        self.saw_error = False
        self._log: list[str] = []

    # -- command -----------------------------------------------------------
    def build_command(self) -> list[str]:
        """Assemble the gimx.exe server command line.

        Argument ORDER matters. GIMX's own docs say: "A --bdaddr, --port or
        --dst argument finishes the current controller options." So --port must
        come LAST, otherwise everything after it is applied to a second,
        non-existent controller.
        """
        cmd = [self.gimx_exe, "--type", self.ctype]
        if self.gimx_config:
            cmd += ["--config", self.gimx_config]
        if self.nograb:
            cmd += ["--nograb"]
        if self.idle_timeout:
            cmd += ["--timeout", str(self.idle_timeout)]
        cmd += ["--src", self.addr]
        cmd += ["--port", self.port]          # must be last
        return cmd

    # -- lifecycle ---------------------------------------------------------
    def start(self, wait_seconds: float = 15.0, quiet: bool = False) -> bool:
        """Launch the server and read its startup output.

        Returns True if GIMX reported detecting the adapter.
        """
        if not Path(self.gimx_exe).is_file():
            print(f"ERROR: gimx.exe not found at {self.gimx_exe}")
            return False

        if session_is_up(quiet=True):
            print(f"A GIMX session is ALREADY reachable at {self.addr}.")
            print("Use 'status' to inspect it, or 'stop' first.")
            return False

        if not self.gimx_config:
            print("WARNING: no config_file for this console profile.")
            print("         Without --config, GIMX forwards NO input at all")
            print("         (not even the physical controller).")

        cmd = self.build_command()
        if not quiet:
            print("Starting GIMX session:")
            print("  " + " ".join(cmd))
            print()

        self.proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1)

        if self.force_normal_priority:
            self._pin_priority()

        deadline = time.time() + wait_seconds
        while time.time() < deadline:
            line = self.proc.stdout.readline() if self.proc.stdout else ""
            if not line:
                if self.proc.poll() is not None:
                    break
                continue
            self._handle_line(line.rstrip(), quiet)
            if self.adapter_detected and self.guide_prompted:
                break

        if self.proc.poll() is not None:
            print("\nGIMX exited during startup. Most likely causes:")
            print("  * the serial port is held by another process")
            print("  * another GIMX session is already running")
            print(f"  * {self.port} does not exist")
            return False

        if not quiet:
            print()
            if self.firmware:
                print(f"  adapter firmware : {self.firmware}")
            if self.baudrate:
                print(f"  baudrate         : {self.baudrate} bps")
            print(f"  listening on UDP : {self.addr}")
            print(f"  serial port      : {self.port}")
            print()
            self._print_auth_banner()
            print(f"Letting the adapter settle for {self.settle:.0f}s ...")
        time.sleep(self.settle)
        return self.adapter_detected

    def _pin_priority(self) -> None:
        """Force Normal priority so GIMX can never starve the OS."""
        if not self.proc:
            return
        try:
            subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 f"(Get-Process -Id {self.proc.pid}).PriorityClass = 'Normal'"],
                capture_output=True, timeout=10)
            print("  [safety] priority pinned to Normal, --nograb enabled")
        except (subprocess.TimeoutExpired, OSError) as exc:
            print(f"  [safety] could not set priority: {exc}")

    def _handle_line(self, line: str, quiet: bool) -> None:
        self._log.append(line)
        if not quiet:
            print(f"  [gimx] {line}")
        if RE_ADAPTER.search(line):
            self.adapter_detected = True
        if RE_GUIDE.search(line):
            self.guide_prompted = True
        m = RE_FIRMWARE.search(line)
        if m:
            self.firmware = m.group(1)
        m = RE_BAUD.search(line)
        if m:
            self.baudrate = m.group(1)
        if RE_ELEVATION.search(line):
            if not quiet:
                print("  [note] That elevation message is HARMLESS here. It "
                      "means GIMX could not take REALTIME priority - which is "
                      "exactly what we want. Do not 'fix' it by running as "
                      "administrator; that can freeze the PC.")
        if RE_ERROR.search(line):
            self.saw_error = True

    def _print_auth_banner(self) -> None:
        print("=" * 66)
        print("  ACTION REQUIRED:  hold the controller's GUIDE button")
        print("                    (the Xbox logo) for 2 SECONDS")
        print()
        print("  Without this the session is NOT authenticated. Events will")
        print("  be accepted and report 'ok' while NOTHING reaches the")
        print("  console. This is the single most common failure.")
        print("=" * 66)

    def stream(self) -> None:
        """Print GIMX output until Ctrl+C. Keeps the session alive."""
        if not self.proc:
            return
        print("\nSession running. Send input from ANOTHER terminal, e.g.:")
        print("    python ../test-controller/test_controller.py press down*3 a")
        print("Press Ctrl+C here to stop the session.\n")
        try:
            if self.proc.stdout:
                for line in self.proc.stdout:
                    print(f"  [gimx] {line.rstrip()}")
        except KeyboardInterrupt:
            print("\nCtrl+C - stopping session ...")
        finally:
            self.stop()

    def stop(self) -> None:
        if not self.proc:
            return
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        print("Session stopped. Serial port released.")
        self.proc = None

    @property
    def log(self) -> list[str]:
        return list(self._log)

    # -- context manager ---------------------------------------------------
    def __enter__(self) -> "GimxSession":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def cmd_start(args: argparse.Namespace) -> int:
    s = GimxSession(args.config, args.console, args.port, args.gimx_config)
    if not s.start():
        return 1
    if args.no_stream:
        print("\nSession started and left running in the background.")
        print("Stop it later with:  python gimx_session.py stop")
        return 0
    s.stream()
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    pids = running_gimx_pids()
    print(f"gimx.exe processes : {pids if pids else 'none'}")
    up = session_is_up(args.config, args.console, quiet=False)
    if up:
        print()
        print("Reachable. Remember this does NOT prove the session was")
        print("authenticated (Guide button) or that input reaches the console.")
        print("Only watching the TV proves that.")
    return 0 if up else 1


def cmd_stop(args: argparse.Namespace) -> int:
    return 0 if stop_all() else 1


def cmd_restart(args: argparse.Namespace) -> int:
    stop_all()
    time.sleep(1)
    return cmd_start(args)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Start / check / stop the GIMX session "
                    "(owns the serial port + Guide-button auth).")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG_PATH),
                    help="path to controls.yaml")
    ap.add_argument("--console", default=None,
                    help="console profile (default: the one marked default)")
    ap.add_argument("--port", default=None, help="serial port, e.g. COM8")
    ap.add_argument("--gimx-config", dest="gimx_config", default=None,
                    help="GIMX .xml config (overrides the profile)")
    ap.add_argument("--no-stream", action="store_true",
                    help="with 'start': don't tail the log, just leave it running")

    sub = ap.add_subparsers(dest="command")
    sub.add_parser("start", help="start a session (then hold GUIDE 2s)")
    sub.add_parser("status", help="is a session running and reachable?")
    sub.add_parser("stop", help="stop any running gimx.exe")
    sub.add_parser("restart", help="stop then start")

    args = ap.parse_args()

    handlers = {"start": cmd_start, "status": cmd_status,
                "stop": cmd_stop, "restart": cmd_restart}
    if args.command not in handlers:
        ap.print_help()
        print("\nTypical use - leave this running in its own terminal:")
        print("    python gimx_session.py start")
        return 1

    try:
        return handlers[args.command](args)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        return 2
    except KeyError as exc:
        print(f"ERROR: {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
