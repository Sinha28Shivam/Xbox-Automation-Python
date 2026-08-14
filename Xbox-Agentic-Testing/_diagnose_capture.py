"""
_diagnose_capture.py - work out WHY the capture feed is black.

    python _diagnose_capture.py

A black frame has several very different causes, and they need different fixes.
This script separates them by probing every capture device on the machine and
saving a frame from each, so you can look at the pictures rather than guess.

THE CAUSES, AND HOW THEY LOOK
-----------------------------
  1. Wrong device index      -> a picture, but of your face. The laptop webcam
                                in a dim room is dark and noisy, and looks
                                remarkably like "no HDMI signal".
  2. RECentral holding it    -> the card cannot be opened at all, or opens and
                                returns nothing. Only one app may hold a UVC
                                device.
  3. Console asleep / no HDMI-> perfectly flat, std ~0.00, 1-2 tones.
  4. HDCP-protected content  -> also flat black, but only for that app.
  5. Grabbed too early       -> the card needs ~1s to lock onto HDMI. The first
                                reads are legitimately blank.
  6. Wrong resolution        -> 1280x720 from a card configured for 1920x1080
                                usually means you are not talking to the card.

WHAT THE NUMBERS MEAN
---------------------
Measured on this rig and recorded in controls.yaml:

    real console content : std ~49,  255 tones,  1920x1080
    no signal            : std 0.00, 1 tone
    dark webcam          : std 1-4,  10-30 tones   <- the trap

std between 1 and 5 with few tones is almost never a console. It is a camera
in a dim room.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for sub in ("core", "tools"):
    p = str(ROOT / sub)
    if p not in sys.path:
        sys.path.insert(0, p)

OUT = ROOT / "artifacts" / "diagnostics"
LINE = "=" * 72


def main() -> int:
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        print(f"Needs opencv + numpy: {exc}")
        return 1

    from config import Config

    cfg = Config.load(ROOT / "config" / "settings.yaml", base=ROOT)
    controls_path = cfg.resolve_path("paths.controls_config", "")

    import yaml
    controls = yaml.safe_load(controls_path.read_text(encoding="utf-8")) or {}
    capture_cfg = controls.get("capture", {}) or {}

    configured_index = int(capture_cfg.get("opencv_index", 1))
    want_w = int(capture_cfg.get("width", 1920))
    want_h = int(capture_cfg.get("height", 1080))
    device_name = capture_cfg.get("device_name", "")

    print(f"\n{LINE}\n  CAPTURE DIAGNOSTIC\n{LINE}")
    print(f"\n  configured index : {configured_index}")
    print(f"  configured size  : {want_w}x{want_h}")
    print(f"  expected device  : {device_name}")

    OUT.mkdir(parents=True, exist_ok=True)

    # -- 1. what does Windows think is attached? ---------------------------
    print(f"\n{LINE}\n  1. VIDEO DEVICES WINDOWS REPORTS\n{LINE}")
    for line in _list_devices():
        print(f"    {line}")

    # -- 2. who is holding a capture device? -------------------------------
    print(f"\n{LINE}\n  2. APPS THAT COULD BE HOLDING THE CARD\n{LINE}")
    hogs = _running_hogs(capture_cfg.get("conflicting_apps") or [])
    if hogs:
        for name in hogs:
            print(f"    RUNNING: {name}")
        print("\n    Only ONE application can hold a capture device.")
        print("    Close these and re-run this diagnostic.")
    else:
        print("    None of the known device-hogging apps are running.")

    # -- 3. probe every index ----------------------------------------------
    print(f"\n{LINE}\n  3. PROBING EACH CAPTURE INDEX\n{LINE}")
    print("\n    idx  opens  resolution   std     tones  verdict")
    print("    " + "-" * 62)

    results = []
    for index in range(4):
        result = _probe(index, want_w, want_h, cv2, np)
        results.append(result)
        if not result["opens"]:
            print(f"    {index:3d}  no     -            -       -      "
                  f"{result['note']}")
            continue
        print(f"    {index:3d}  yes    {result['size']:11s}  "
              f"{result['std']:6.2f}  {result['tones']:5d}  "
              f"{result['verdict']}")
        if result["path"]:
            print(f"         saved: {result['path']}")

    # -- 4. conclusion -----------------------------------------------------
    print(f"\n{LINE}\n  DIAGNOSIS\n{LINE}\n")
    _diagnose(results, configured_index, want_w, want_h, hogs)

    print(f"\n  Frames saved to: {OUT}")
    print("  OPEN THEM. A picture settles in one second what statistics")
    print("  only suggest - especially whether you are looking at the")
    print("  console or at your own webcam.\n")
    return 0


# ===========================================================================
def _probe(index: int, want_w: int, want_h: int, cv2, np) -> dict:
    """Open one index, wait for it to lock on, and describe what arrives."""
    import time

    out = {"index": index, "opens": False, "size": "", "std": 0.0,
           "tones": 0, "verdict": "", "note": "", "path": None,
           "width": 0, "height": 0}

    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        out["note"] = "cannot open (absent, or held by another app)"
        return out

    out["opens"] = True
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, want_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, want_h)

    # The card needs ~1s to lock onto HDMI; the first reads are genuinely
    # blank. Poll rather than trusting a fixed frame count, and keep the BEST
    # frame seen - grabbing once and giving up is how a working card gets
    # misdiagnosed as dead.
    best = None
    best_std = -1.0
    deadline = time.time() + 3.0
    while time.time() < deadline:
        ok, frame = cap.read()
        if ok and frame is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            std = float(gray.std())
            if std > best_std:
                best_std, best = std, frame
            if std > 5.0:          # clearly real content; no need to keep going
                break
        time.sleep(0.1)
    cap.release()

    if best is None:
        out["note"] = "opened but returned no frame"
        return out

    gray = cv2.cvtColor(best, cv2.COLOR_BGR2GRAY)
    h, w = best.shape[:2]
    out.update({
        "width": w, "height": h,
        "size": f"{w}x{h}",
        "std": float(gray.std()),
        "tones": int(len(np.unique(gray[::8, ::8]))),
        "mean": float(gray.mean()),
    })
    out["verdict"] = _classify(out, want_w, want_h)

    path = OUT / f"index-{index}.png"
    if cv2.imwrite(str(path), best):
        out["path"] = str(path)
    return out


def _classify(r: dict, want_w: int, want_h: int) -> str:
    """Name what this frame most likely is."""
    std, tones = r["std"], r["tones"]

    if std < 1.0 or tones < 4:
        return "FLAT - no signal"
    if std < 5.0 and tones < 40:
        # The trap from docs 10: a webcam in a dim room passes a naive
        # "is it blank?" check while showing nothing useful.
        return "near-black (dark webcam?)"
    if r["width"] != want_w or r["height"] != want_h:
        return f"content, but {r['size']} not {want_w}x{want_h}"
    return "REAL CONTENT"


def _diagnose(results: list[dict], configured: int, want_w: int, want_h: int,
              hogs: list[str]) -> None:
    """Turn the probe results into an ordered list of things to do."""
    good = [r for r in results
            if r["opens"] and r["std"] >= 5.0 and r["tones"] >= 40]
    configured_result = next(
        (r for r in results if r["index"] == configured), None)

    if good:
        best = max(good, key=lambda r: r["std"])
        if best["index"] != configured:
            print(f"  >> LIKELY CAUSE: WRONG INDEX.\n")
            print(f"     Index {configured} is configured, but index "
                  f"{best['index']} has real content")
            print(f"     ({best['size']}, std {best['std']:.1f}, "
                  f"{best['tones']} tones).")
            print(f"\n     Check artifacts/diagnostics/index-{best['index']}.png "
                  f"is the console,")
            print(f"     then set in controls.yaml:\n")
            print(f"         capture:\n           opencv_index: {best['index']}")
            return

        if best["width"] != want_w:
            print(f"  >> Index {configured} has content but at "
                  f"{best['size']}, not {want_w}x{want_h}.")
            print(f"\n     Either the console is outputting a lower resolution,")
            print(f"     or this is not the capture card. Open")
            print(f"     artifacts/diagnostics/index-{configured}.png to see which.")
            return

        print(f"  >> Capture looks HEALTHY on index {configured}: "
              f"{best['size']}, std {best['std']:.1f}.")
        print(f"     If the framework still reports black, the console screen")
        print(f"     itself may genuinely be dark right now.")
        return

    # Nothing anywhere had real content.
    print("  >> NO device produced real content.\n")

    if hogs:
        print(f"     MOST LIKELY: {', '.join(hogs)} is holding the capture card.")
        print(f"     Only one application can open a capture device at a time.")
        print(f"     Close it, then re-run this diagnostic.\n")

    if configured_result and configured_result["opens"]:
        std = configured_result["std"]
        if std < 1.0:
            print(f"     Index {configured} opens but is PERFECTLY FLAT "
                  f"(std {std:.2f}).")
            print(f"     That is a true no-signal frame - the card is fine,")
            print(f"     nothing is arriving on HDMI. In order:\n")
            print(f"       1. Is the console powered on and AWAKE (not standby)?")
            print(f"       2. Is HDMI in the card's IN port (not OUT)?")
            print(f"       3. Is a protected app (Netflix etc.) on screen?")
            print(f"          HDCP content always captures black.")
            print(f"       4. Try the HDMI cable in a different port.")
        else:
            print(f"     Index {configured} is near-black "
                  f"(std {std:.2f}, {configured_result['tones']} tones).")
            print(f"     That is too noisy to be 'no signal' and too dark to")
            print(f"     be a dashboard - it looks like a WEBCAM in a dim room.")
            print(f"     Open artifacts/diagnostics/index-{configured}.png "
                  f"and check.")
    else:
        print(f"     Index {configured} could not be opened at all.")
        print(f"     Either the card is unplugged, or another app holds it.")


def _list_devices() -> list[str]:
    """Ask Windows what video devices exist. ffmpeg first, then PnP."""
    try:
        res = subprocess.run(
            ["ffmpeg", "-hide_banner", "-list_devices", "true",
             "-f", "dshow", "-i", "dummy"],
            capture_output=True, text=True, timeout=20)
        lines = [l.strip() for l in (res.stderr or "").splitlines()
                 if '"' in l and ("video" in l.lower() or "alternative" not in l.lower())]
        if lines:
            return lines[:12]
    except (subprocess.TimeoutExpired, OSError, FileNotFoundError):
        pass

    try:
        res = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-PnpDevice -Class Camera,Image,Media -PresentOnly | "
             "Select-Object -ExpandProperty FriendlyName"],
            capture_output=True, text=True, timeout=20)
        return [l.strip() for l in (res.stdout or "").splitlines() if l.strip()]
    except (subprocess.TimeoutExpired, OSError):
        return ["(could not enumerate devices)"]


def _running_hogs(names: list[str]) -> list[str]:
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
        key = name.lower().split()[0]
        if any(r.startswith(key) for r in running if r):
            hits.append(name)
    return hits


if __name__ == "__main__":
    sys.exit(main())
