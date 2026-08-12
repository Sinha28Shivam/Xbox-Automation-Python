"""
capture.py - Screen capture for the automation framework.

ANSWER TO "do I need to run AVerMedia/RECentral?"
-------------------------------------------------
NO. You do not launch RECentral, and you do not start anything by hand.
The framework opens the capture card itself, automatically.

RECentral is AVerMedia's consumer recording GUI. The card is a standard UVC
device, so we talk to it directly. RECentral is only useful as an eyeball check
that the console is outputting video - and it must be CLOSED while automating,
because only ONE application can hold a capture device at a time.

    preflight()  checks all of this for you and says exactly what is wrong.

USAGE
-----
    # one-liner check before a test run
    python capture.py preflight

    # save a screenshot of the console right now
    python capture.py grab --out shot.png

    # live stats, useful while wiggling cables
    python capture.py watch

From Python:

    from capture import ScreenCapture, preflight

    if not preflight().ok:
        sys.exit("capture not ready")

    with ScreenCapture() as cam:          # opens the card automatically
        frame = cam.grab()                # numpy BGR array
        cam.wait_for_stable_screen()      # wait until animation settles
        changed = cam.changed_since(frame)

DESIGN NOTE
-----------
`grab()` refuses to silently hand back a blank frame. A black image is still a
valid image, and a verification layer built on blank frames would "pass" forever
without ever seeing the console. Blank frames raise or return None by design.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import cv2
    import numpy as np
except ImportError:
    sys.exit("Needs opencv + numpy:  pip install opencv-python numpy")

try:
    import yaml
except ImportError:
    sys.exit("PyYAML is required:  pip install pyyaml")

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "controls.yaml"


def load_capture_config(path: Path | str = CONFIG_PATH) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    cfg = data.get("capture") or {}
    cfg.setdefault("device_name", "AVerMedia ExtremeCap UVC")
    cfg.setdefault("opencv_index", 1)
    cfg.setdefault("width", 1920)
    cfg.setdefault("height", 1080)
    cfg.setdefault("fps", 30)
    cfg.setdefault("warmup_frames", 3)
    cfg.setdefault("sync_timeout", 5.0)
    cfg.setdefault("blank_std_threshold", 1.0)
    cfg.setdefault("conflicting_apps", ["RECentral 4", "OBS", "Teams"])
    return cfg


# --------------------------------------------------------------------------
# Frame helpers
# --------------------------------------------------------------------------
def frame_stats(frame: "np.ndarray") -> dict[str, float]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    return {
        "mean": float(gray.mean()),
        "std": float(gray.std()),
        "tones": float(len(np.unique(gray[::8, ::8]))),
    }


def is_blank(frame: "np.ndarray", threshold: float = 1.0) -> bool:
    """A flat image means no signal - NOT a dark game scene.

    Even a night-time game scene has gradients and noise. A true no-signal
    frame is one single value everywhere (std 0.0), so this is safe.
    """
    s = frame_stats(frame)
    return s["std"] < threshold or s["tones"] < 4


def difference(a: "np.ndarray", b: "np.ndarray") -> float:
    """Mean absolute pixel difference between two frames (0 = identical)."""
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY) if a.ndim == 3 else a
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY) if b.ndim == 3 else b
    if ga.shape != gb.shape:
        gb = cv2.resize(gb, (ga.shape[1], ga.shape[0]))
    return float(cv2.absdiff(ga, gb).mean())


# --------------------------------------------------------------------------
# Preflight
# --------------------------------------------------------------------------
@dataclass
class Preflight:
    ok: bool = False
    device_present: bool = False
    device_opens: bool = False
    has_signal: bool = False
    conflicts: list[str] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)

    def report(self) -> None:
        print("=" * 64)
        print("CAPTURE PREFLIGHT")
        print("=" * 64)
        tick = lambda b: "OK  " if b else "FAIL"
        print(f"  [{tick(self.device_present)}] capture card connected")
        print(f"  [{tick(self.device_opens)}] device can be opened "
              f"(nothing else holding it)")
        print(f"  [{tick(self.has_signal)}] console video signal present")
        if self.conflicts:
            print(f"\n  Apps that may steal the device: "
                  f"{', '.join(self.conflicts)}")
        if self.messages:
            print()
            for m in self.messages:
                print(f"  {m}")
        print("=" * 64)
        print("READY - the framework will capture automatically."
              if self.ok else "NOT READY - fix the above first.")
        print("=" * 64)


def running_conflicts(names: list[str]) -> list[str]:
    """Which known device-hogging apps are running right now."""
    try:
        res = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-Process | Select-Object -ExpandProperty ProcessName"],
            capture_output=True, text=True, timeout=15)
    except (subprocess.TimeoutExpired, OSError):
        return []
    running = {p.strip().lower() for p in (res.stdout or "").splitlines()}
    hits = []
    for name in names:
        # "RECentral 4" -> process name is "RECentral 4"; match loosely
        key = name.lower()
        if any(key.startswith(r) or r.startswith(key.split()[0])
               for r in running if r):
            hits.append(name)
    return hits


def preflight(config_path: Path | str = CONFIG_PATH,
              quiet: bool = False) -> Preflight:
    """Check everything needed for automatic capture. Starts nothing."""
    cfg = load_capture_config(config_path)
    out = Preflight()

    # 1. Is the card physically there?
    try:
        res = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-PnpDevice -PresentOnly | "
             f"Where-Object {{ $_.InstanceId -match '{cfg.get('usb_id','VID_07CA')}' }} | "
             "Select-Object -ExpandProperty FriendlyName"],
            capture_output=True, text=True, timeout=20)
        out.device_present = bool((res.stdout or "").strip())
    except (subprocess.TimeoutExpired, OSError):
        out.device_present = False

    if not out.device_present:
        out.messages.append("Capture card not detected. Check its USB cable.")
        if not quiet:
            out.report()
        return out

    # 2. Who might steal it? (informational - RECentral GUI is the usual one)
    out.conflicts = running_conflicts(cfg["conflicting_apps"])

    # 3. Can we open it, and is there a picture?
    cam = ScreenCapture(cfg)
    try:
        cam.open()
        out.device_opens = True
        frame = cam.grab(allow_blank=True)
        if frame is None:
            out.messages.append("Device opened but returned no frame.")
        else:
            out.has_signal = not is_blank(frame, cfg["blank_std_threshold"])
            s = frame_stats(frame)
            out.messages.append(
                f"frame {frame.shape[1]}x{frame.shape[0]}  "
                f"std={s['std']:.2f}  tones={int(s['tones'])}")
            if not out.has_signal:
                out.messages.append(
                    "Frame is FLAT (std 0) = NO HDMI SIGNAL. The device is")
                out.messages.append(
                    "fine; nothing is coming in. Check, in order:")
                out.messages.append(
                    "  1. Console powered on and awake (not in standby)")
                out.messages.append(
                    "  2. HDMI from console -> card's IN port (not OUT)")
                out.messages.append(
                    "  3. HDCP: protected apps (Netflix) capture black")
    except RuntimeError as exc:
        out.device_opens = False
        out.messages.append(f"Could not open the device: {exc}")
        out.messages.append(
            "This usually means another app holds it - close RECentral 4.")
    finally:
        cam.close()

    out.ok = out.device_present and out.device_opens and out.has_signal
    if not quiet:
        out.report()
    return out


# --------------------------------------------------------------------------
# The capture object
# --------------------------------------------------------------------------
class ScreenCapture:
    """Opens the capture card and hands out frames.

    Opening is AUTOMATIC - no external application is needed or wanted.
    Keep one instance alive for a test run: the first frame costs ~700 ms
    (device open) but every frame after that is ~1 ms.
    """

    def __init__(self, config: dict[str, Any] | None = None,
                 config_path: Path | str = CONFIG_PATH):
        self.cfg = config or load_capture_config(config_path)
        self.index = int(self.cfg["opencv_index"])
        self.width = int(self.cfg["width"])
        self.height = int(self.cfg["height"])
        self.blank_threshold = float(self.cfg["blank_std_threshold"])
        self.warmup = int(self.cfg["warmup_frames"])
        self.sync_timeout = float(self.cfg.get("sync_timeout", 5.0))
        self.synced = False
        self.cap: "cv2.VideoCapture | None" = None

    def open(self) -> None:
        """Open the card and wait until it is actually delivering pixels.

        MEASURED on this hardware: after opening, the first ~2 reads come back
        perfectly flat (std 0.00) while the card locks onto the HDMI signal.
        Real content appears at ~1.05 s.

        A fixed "discard N frames" warmup is fragile, so we POLL until a
        non-blank frame arrives. This matters a lot: without it an immediate
        grab looks identical to "no HDMI signal", and you go hunting for a
        hardware fault that does not exist. That exact confusion happened
        during development.
        """
        if self.cap is not None:
            return
        cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            raise RuntimeError(
                f"could not open capture device index {self.index}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap = cap

        for _ in range(self.warmup):        # discard known-bad first reads
            cap.read()

        deadline = time.time() + self.sync_timeout
        while time.time() < deadline:
            ok, frame = cap.read()
            if ok and frame is not None and not is_blank(
                    frame, self.blank_threshold):
                self.synced = True
                return
            time.sleep(0.1)
        # Still flat after the timeout: probably a genuinely absent signal.
        self.synced = False

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.synced = False

    def grab(self, allow_blank: bool = False) -> "np.ndarray | None":
        """One frame. Returns None on a blank frame unless allow_blank.

        Refusing to return blank frames is deliberate: a verification layer fed
        black images would report success forever without ever seeing the
        console.
        """
        if self.cap is None:
            self.open()
        assert self.cap is not None
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return None
        if not allow_blank and is_blank(frame, self.blank_threshold):
            return None
        return frame

    def grab_or_raise(self) -> "np.ndarray":
        frame = self.grab()
        if frame is None:
            raise RuntimeError(
                "No usable frame: the feed is blank (no HDMI signal) or the "
                "device is held by another app such as RECentral 4.")
        return frame

    def wait_for_stable_screen(self, timeout: float = 10.0,
                               settle: float = 0.4,
                               threshold: float = 0.5) -> "np.ndarray | None":
        """Wait until the picture stops changing (animation finished).

        Replaces guessy sleep() calls: instead of waiting a fixed time and
        hoping, we wait until consecutive frames stop differing.
        """
        deadline = time.time() + timeout
        prev = self.grab(allow_blank=True)
        stable_since = None
        while time.time() < deadline:
            time.sleep(0.1)
            cur = self.grab(allow_blank=True)
            if cur is None or prev is None:
                prev = cur
                continue
            if difference(prev, cur) < threshold:
                stable_since = stable_since or time.time()
                if time.time() - stable_since >= settle:
                    return cur
            else:
                stable_since = None
            prev = cur
        return prev

    def changed_since(self, before: "np.ndarray",
                      threshold: float = 0.5) -> bool:
        after = self.grab(allow_blank=True)
        if after is None:
            return False
        return difference(before, after) >= threshold

    def save(self, path: Path | str, frame: "np.ndarray | None" = None) -> bool:
        frame = frame if frame is not None else self.grab(allow_blank=True)
        if frame is None:
            return False
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        return bool(cv2.imwrite(str(path), frame))

    def __enter__(self) -> "ScreenCapture":
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def cmd_preflight(args: argparse.Namespace) -> int:
    return 0 if preflight(args.config).ok else 1


def cmd_grab(args: argparse.Namespace) -> int:
    with ScreenCapture(config_path=args.config) as cam:
        frame = cam.grab(allow_blank=True)
        if frame is None:
            print("No frame returned.")
            return 1
        s = frame_stats(frame)
        blank = is_blank(frame, cam.blank_threshold)
        print(f"{frame.shape[1]}x{frame.shape[0]}  std={s['std']:.2f}  "
              f"tones={int(s['tones'])}  -> "
              f"{'BLANK (no signal)' if blank else 'has content'}")
        cam.save(args.out, frame)
        print(f"saved: {args.out}")
        return 1 if blank else 0


def cmd_watch(args: argparse.Namespace) -> int:
    """Live stats - handy while checking cables or waking the console."""
    print("Watching the feed. Ctrl+C to stop.\n")
    with ScreenCapture(config_path=args.config) as cam:
        prev = None
        try:
            while True:
                frame = cam.grab(allow_blank=True)
                if frame is None:
                    print("  no frame")
                else:
                    s = frame_stats(frame)
                    d = difference(prev, frame) if prev is not None else 0.0
                    state = ("BLANK (no signal)"
                             if is_blank(frame, cam.blank_threshold)
                             else "has content")
                    print(f"  std={s['std']:6.2f}  tones={int(s['tones']):3d}  "
                          f"delta={d:6.3f}  {state}")
                    prev = frame
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("\nstopped")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Console screen capture. The card is opened AUTOMATICALLY "
                    "- you do not need to run RECentral or any AVerMedia app.")
    ap.add_argument("--config", default=str(CONFIG_PATH))
    sub = ap.add_subparsers(dest="command")

    sub.add_parser("preflight", help="check the card, the lock and the signal")

    g = sub.add_parser("grab", help="save one screenshot")
    g.add_argument("--out", default="screenshot.png")

    sub.add_parser("watch", help="print live frame stats once per second")

    args = ap.parse_args()
    handlers = {"preflight": cmd_preflight, "grab": cmd_grab,
                "watch": cmd_watch}
    if args.command not in handlers:
        ap.print_help()
        print("\nStart with:  python capture.py preflight")
        return 1
    try:
        return handlers[args.command](args)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"ERROR: {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
