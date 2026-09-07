"""MANUAL Magic Marker tuner - you pick the values, the script performs them.

Sequence performed (the real way the mechanic is played):
    1. hold RT                        -> magic marker opens, time slows
    2. push the stick toward the glow -> moves the marker CURSOR (8 directions,
                                         diagonals included)
    3. hold A                         -> ink grabs that spot on the earth
    4. drag the stick (default UP)    -> the stroke grows into a pillar
    5. release A, then release RT     -> stroke commits, time resumes

WHY A HOLDER THREAD: one gimx.exe call costs ~250ms and carries a state, not a
latch. Sending RT then A then the stick as separate calls means RT has already
lapsed before the stroke starts and the marker closes. So a background thread
re-asserts the WHOLE combination (RT + A + stick) every tick, using multiple
--event flags in a single gimx.exe invocation.

WHY RT IS 32767: hardware-verified. On XOnePad the trigger axis spans
0..32767 like a stick, NOT 0..255. r2(255) is ~0.8% of a pull, which gimx.exe
accepts but the console ignores. Threshold sits between 512 and 1023.

-----------------------------------------------------------------------------
USAGE - interactive (just run it and answer the prompts):
    python _manual_marker.py

USAGE - one shot, all values on the command line:
    python _manual_marker.py --aim down-right --aim-time 0.1 --draw up \
        --draw-time 1.8 --strength 1.0

USAGE - sweep several aim directions in one go, pausing between each:
    python _manual_marker.py --aim right,down-right,down --aim-time 0.1

Every run saves _mm_<n>_<stage>.png frames and prints frame-difference numbers
so you can tell what changed even without looking at the TV.

-----------------------------------------------------------------------------
MEASURED FROM A REAL RUN (see MARKER_FINDINGS.md):
  * Max stood at (1161, 875) and did NOT move while RT was held, so the left
    stick moves the CURSOR here, not the character.
  * When the marker opens, the cursor reticle rests at about (974, 530) -
    up in the tree canopy, roughly up-LEFT of Max.
  * The glowing earth is the amber pool at (1246, 891): only dx=+85, dy=+16
    from Max, i.e. essentially LEVEL-RIGHT and very close.
  * Therefore the cursor has to travel from (974,530) to (1246,891):
    dx=+272, dy=+361  ->  aim DOWN-RIGHT, not up-right.
  * '--aim up-right' pushed the cursor further AWAY from the target, which is
    why the ink landed in the wrong place.

WHICH STICK MOVES THE CURSOR
  lstick, for BOTH modes. It is the confirmed working aiming stick for
  drawing, and rstick was tried for destroying and did NOT work either, so
  there is no reason to keep them different. Destroy now uses byte-for-byte
  the same aim as the proven draw command: lstick, --aim-time 0.1.

  The reticle only shifts ~9px during the aim stage, which looks like
  nothing, yet the stroke still lands correctly - so that pixel measurement
  is simply a bad proxy for where the ink goes. Do not "fix" it.

  Override with --aim-stick / --draw-stick if you want to experiment.

CONFIRMED WORKING - both operations, on hardware:
    # draw
    python _manual_marker.py --aim down-right --aim-time 0.4 --draw up \
        --draw-time 1.8 --strength 1.0
    # destroy
    python _manual_marker.py --mode destroy --aim down-right --aim-time 0.4 \
        --strength 1.0

STRENGTH MATTERS MORE THAN TIME. Measured gain is ~450 px/s at strength 1.0,
but only ~17 px/s at strength 0.4 - a reduced deflection nearly freezes the
cursor. Always keep --strength 1.0 unless you specifically want fine nudges.
0.1s (~40px) is enough to DRAW, because the ink snaps to nearby glowing
earth, but not to DESTROY, which needs the cursor physically on the pillar.

-----------------------------------------------------------------------------
DESTROY MODE (--mode destroy)
  Removes an existing drawing instead of creating one:
      1. hold RT                  -> marker opens
      2. aim at the PILLAR        -> same stick aiming as drawing
      3. press X (gimx "square")  -> the drawing is erased
      4. release RT
  The X-prompt icon visible next to a finished pillar in the captured frames
  is what confirms X is the erase button.

      python _manual_marker.py --mode destroy --aim down-right

  This now uses exactly the same aiming as the proven draw command
  (lstick, --aim-time 0.1). If X erases nothing, escalate:
      --hold-aim        keep the stick deflected while X is pressed, so the
                        cursor cannot drift back to rest first
      --repeat-x 3      press X several times
      --x-hold 0.6      hold X longer

  Note: aim at the PILLAR you just made, which is not necessarily the same
  direction as the glowing earth you drew it on - the camera PANS after a
  successful draw, so the old aim direction is stale.
-----------------------------------------------------------------------------
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for sub in ("core", "tools", "agents", "graph"):
    p = str(ROOT / sub)
    if p not in sys.path:
        sys.path.insert(0, p)

import cv2                                          # noqa: E402
import numpy as np                                  # noqa: E402
from adapters import HardwareBridge                 # noqa: E402
from config import Config, load_dotenv_if_present   # noqa: E402

GIMX = r"C:\Program Files\GIMX\gimx.exe"
DST = "127.0.0.1:51914"
CTYPE = "XOnePad"

FULL = 32767
RT_DEFAULT = 32767

# Screen-space direction -> stick vector, as fractions of full deflection.
# Screen +y is DOWN and the stick's +y is DOWN too, so "up" is negative y.
# Diagonals use 0.7071 so the total deflection still equals 1.0.
D = 0.7071
DIRECTIONS: dict[str, tuple[float, float]] = {
    "up":         (0.0, -1.0),
    "down":       (0.0,  1.0),
    "left":       (-1.0, 0.0),
    "right":      (1.0,  0.0),
    "up-left":    (-D, -D),
    "up-right":   (D,  -D),
    "down-left":  (-D,  D),
    "down-right": (D,   D),
    "none":       (0.0,  0.0),
}
ALIASES = {
    "u": "up", "d": "down", "l": "left", "r": "right",
    "ul": "up-left", "ur": "up-right", "dl": "down-left", "dr": "down-right",
    "upleft": "up-left", "upright": "up-right",
    "downleft": "down-left", "downright": "down-right",
    "left-up": "up-left", "right-up": "up-right",
    "left-down": "down-left", "right-down": "down-right",
    "centre": "none", "center": "none", "n": "none", "": "none",
}


def resolve_dir(name: str) -> str:
    key = (name or "").strip().lower().replace("_", "-").replace(" ", "-")
    key = ALIASES.get(key, key)
    if key not in DIRECTIONS:
        raise SystemExit(
            f"Unknown direction '{name}'.\nValid: {', '.join(DIRECTIONS)}")
    return key


def stick_xy(direction: str, strength: float) -> tuple[int, int]:
    fx, fy = DIRECTIONS[direction]
    s = max(0.0, min(1.0, float(strength)))
    return int(round(fx * FULL * s)), int(round(fy * FULL * s))


def send(events: list[tuple[str, int]]) -> None:
    """One gimx.exe call carrying ALL the events, so they apply together."""
    if not events:
        return
    cmd = [GIMX, "--type", CTYPE]
    for control, value in events:
        cmd += ["--event", f"{control}({value})"]
    cmd += ["--dst", DST]
    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    except Exception as exc:                       # keep the run alive
        print(f"   ! gimx call failed: {exc}")


class Holder(threading.Thread):
    """Re-asserts the pad state continuously so no hold is ever dropped."""

    def __init__(self) -> None:
        super().__init__(daemon=True)
        self._state: list[tuple[str, int]] = []
        self._stop = threading.Event()
        self._lock = threading.Lock()

    def set_state(self, events: list[tuple[str, int]]) -> None:
        with self._lock:
            self._state = list(events)

    def run(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                state = list(self._state)
            if state:
                send(state)
            else:
                time.sleep(0.02)

    def stop(self) -> None:
        self._stop.set()


def ask(prompt: str, default: str) -> str:
    try:
        raw = input(f"{prompt} [{default}]: ").strip()
    except EOFError:
        return default
    return raw or default


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Manually tune the Magic Marker aim + draw.")
    ap.add_argument("--aim", default=None,
                    help="Direction(s) from Max toward the glowing earth. "
                         "Comma-separate to try several: 'up-right,right'. "
                         f"Valid: {', '.join(DIRECTIONS)}")
    ap.add_argument("--aim-time", type=float, default=None,
                    help="Seconds to hold the stick while moving the cursor.")
    ap.add_argument("--draw", default=None,
                    help="Direction of the actual stroke (usually 'up').")
    ap.add_argument("--draw-time", type=float, default=None,
                    help="Seconds to drag the stroke.")
    ap.add_argument("--strength", type=float, default=None,
                    help="Stick deflection 0.0-1.0 (lower = finer aiming).")
    ap.add_argument("--grab-time", type=float, default=0.45,
                    help="Seconds to hold A before dragging (ink anchor).")
    ap.add_argument("--rt", type=int, default=RT_DEFAULT,
                    help="RT analog value. 32767 = full pull. <1023 does "
                         "NOT open the marker.")
    ap.add_argument("--settle", type=float, default=1.8,
                    help="Seconds to wait after RT for the marker to open.")
    ap.add_argument("--mode", default="draw",
                    choices=["draw", "destroy", "calib"],
                    help="draw = hold A and drag a stroke; "
                         "destroy = aim at an existing drawing and press X; "
                         "calib = measure cursor travel in px/s (no drawing).")
    ap.add_argument("--x-hold", type=float, default=0.25,
                    help="Seconds to hold X when destroying.")
    ap.add_argument("--hold-aim", action="store_true",
                    help="Destroy mode: keep the aim stick DEFLECTED while X "
                         "is pressed, instead of re-centring it first. Use "
                         "this if the cursor snaps back to rest before X "
                         "lands, so X erases nothing.")
    ap.add_argument("--repeat-x", type=int, default=1,
                    help="Press X this many times (some drawings need more "
                         "than one hit).")
    ap.add_argument("--aim-stick", default="lstick",
                    choices=["rstick", "lstick"],
                    help="Which stick moves the cursor. Always lstick: it is "
                         "the PROVEN aiming stick for drawing, and rstick "
                         "did not work for destroying either.")
    ap.add_argument("--draw-stick", default="lstick",
                    choices=["rstick", "lstick"],
                    help="Which stick drags the stroke. Default lstick, "
                         "which is the proven working value.")
    return ap.parse_args()


def interactive_fill(args: argparse.Namespace) -> None:
    """Prompt for anything not supplied on the command line."""
    destroying = args.mode in ("destroy", "calib")
    if args.mode == "calib":
        # Calibration sweeps its own hold times and never draws.
        args.aim = args.aim or "down-right"
        args.aim_time = args.aim_time or 0.1
        args.strength = 1.0 if args.strength is None else args.strength
        args.draw, args.draw_time = args.draw or "up", 0.0
        return
    needed = (args.aim is None or args.aim_time is None
              or args.strength is None
              or (not destroying and (args.draw is None
                                      or args.draw_time is None)))
    if needed:
        print("=" * 74)
        print(f" MANUAL MAGIC MARKER TUNER  -  mode: {args.mode.upper()}")
        print("=" * 74)
        if destroying:
            print(" Aim at the EXISTING DRAWING you want to erase, then X.")
        else:
            print(" Stand Max NEXT TO the glowing earth first.")
        print(" Directions are in SCREEN space, as seen from Max:")
        print("   up   up-right   right   down-right   down   down-left"
              "   left   up-left")
        print(" Press Enter to accept the [default].\n")

    if args.aim is None:
        # down-right, not up-right: measured cursor rest (974,530) is ABOVE
        # and LEFT of the real glow at (1246,891). See MARKER_FINDINGS.md.
        args.aim = ask(" Aim direction toward the glow", "down-right")
    if args.aim_time is None:
        # 0.4 at strength 1.0 is CONFIRMED for both drawing and destroying.
        # Measured gain is ~450 px/s at full deflection, so 0.4s ~= 180px.
        # 0.1s (~40px) only worked for drawing, where the ink snaps to nearby
        # glowing earth; destroying has no snap and needs real travel.
        args.aim_time = float(ask(" Aim hold seconds", "0.4"))
    if args.strength is None:
        args.strength = float(ask(" Aim stick strength 0.0-1.0", "1.0"))
    if destroying:
        # No stroke is drawn when erasing, so keep the draw values inert.
        args.draw = args.draw or "up"
        args.draw_time = 0.0 if args.draw_time is None else args.draw_time
    else:
        if args.draw is None:
            args.draw = ask(" Draw/stroke direction", "up")
        if args.draw_time is None:
            args.draw_time = float(ask(" Draw seconds", "1.8"))


def run_calib(hw, holder, args, aim_dir: str) -> None:
    """Measure how far the cursor travels per second of stick deflection.

    Holds RT once, then applies the stick for increasing durations WITHOUT
    ever re-centring, tracking the reticle after each step. This answers the
    only question that matters for aiming: px per second at this strength.
    """
    cap = hw.capture()
    ax, ay = stick_xy(aim_dir, args.strength)
    AX, AY = f"{args.aim_stick} x", f"{args.aim_stick} y"
    ZERO = [("lstick x", 0), ("lstick y", 0),
            ("rstick x", 0), ("rstick y", 0)]

    def shot(name, flush=3):
        for _ in range(flush):
            cap.grab(allow_blank=True)
        f = cap.grab(allow_blank=True)
        cv2.imwrite(str(ROOT / f"_cal_{name}.png"), f)
        return f

    def pale(img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        return cv2.morphologyEx(
            cv2.inRange(hsv, np.array([0, 0, 170]), np.array([180, 110, 255])),
            cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    def find(img, base):
        m = cv2.bitwise_and(pale(img), cv2.bitwise_not(base))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        best = None
        for c in cnts:
            a = cv2.contourArea(c)
            if a < 700 or a > 40000:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            if bh == 0 or not 0.45 < bw / float(bh) < 2.2:
                continue
            if a / float(bw * bh) < 0.45:
                continue
            mo = cv2.moments(c)
            if mo["m00"] and (best is None or a > best[0]):
                best = (a, mo["m10"] / mo["m00"], mo["m01"] / mo["m00"])
        return best

    print(f"\n{'=' * 74}")
    print(f" CALIBRATION  aim={aim_dir} strength={args.strength} "
          f"stick={args.aim_stick}")
    print("=" * 74)

    holder.set_state([])
    send([("r2", 0), ("cross", 0), ("square", 0)] + ZERO)
    time.sleep(1.0)
    closed = shot("0closed")
    base = pale(closed)

    holder.set_state([("r2", args.rt)] + ZERO)
    time.sleep(args.settle)
    start = find(shot("1open"), base)
    if start is None:
        print("  !! cursor not found with the marker open - cannot calibrate.")
        holder.set_state([])
        send([("r2", 0)] + ZERO)
        return
    print(f"  cursor rest: ({start[1]:.1f}, {start[2]:.1f})")
    print(f"  {'held(s)':>8} {'cursor':>20} {'moved':>10} {'px/s':>9}")

    steps = [0.25, 0.5, 1.0, 2.0]
    elapsed = 0.0
    prev = start
    for held in steps:
        # Apply the stick for this slice, never re-centring in between, so
        # total elapsed deflection time accumulates.
        holder.set_state([("r2", args.rt), (AX, ax), (AY, ay)])
        time.sleep(held)
        holder.set_state([("r2", args.rt), (AX, ax), (AY, ay)])
        elapsed += held
        cur = find(shot(f"2held{elapsed:.2f}", flush=1), base)
        if cur is None:
            print(f"  {elapsed:8.2f} {'lost':>20}")
            continue
        tdx, tdy = cur[1] - start[1], cur[2] - start[2]
        tot = (tdx * tdx + tdy * tdy) ** 0.5
        print(f"  {elapsed:8.2f} ({cur[1]:7.1f},{cur[2]:7.1f}) "
              f"{tot:9.1f} {tot / elapsed:8.1f}")
        prev = cur

    holder.set_state([])
    send([("r2", 0), ("cross", 0), ("square", 0)] + ZERO)

    tdx, tdy = prev[1] - start[1], prev[2] - start[2]
    tot = (tdx * tdx + tdy * tdy) ** 0.5
    print(f"\n  TOTAL travel after {elapsed:.2f}s: {tot:.1f}px "
          f"(dx={tdx:+.1f} dy={tdy:+.1f})")
    if tot < 40:
        print("  VERDICT: the cursor is essentially STUCK on "
              f"{args.aim_stick}.")
        print("           Aiming cannot work this way. Either the game")
        print("           auto-snaps the ink (so drawing succeeds without")
        print("           real aiming), or the cursor is on another input.")
        print(f"           Try: --mode calib --aim-stick "
              f"{'rstick' if args.aim_stick == 'lstick' else 'lstick'}")
    else:
        print(f"  GAIN: ~{tot / elapsed:.0f} px/s at strength "
              f"{args.strength} on {args.aim_stick}")
        print(f"  To move N px, hold roughly N/{tot / elapsed:.0f} seconds.")
    print("  Frames saved as _cal_*.png")


def run_once(hw, holder, args, aim_dir: str, tag: str) -> None:
    cap = hw.capture()
    diff = hw.capture_functions()["difference"]

    def grab(stage: str, flush: int = 3):
        # The capture pipeline LAGS: a frame grabbed right after an action can
        # still show the PREVIOUS screen. Flush a few frames before measuring.
        for _ in range(flush):
            cap.grab(allow_blank=True)
        frame = cap.grab(allow_blank=True)
        cv2.imwrite(str(ROOT / f"_mm_{tag}_{stage}.png"), frame)
        mean = float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean())
        print(f"      [{stage:<10}] brightness={mean:6.2f}")
        return frame

    def pale(img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        return cv2.morphologyEx(
            cv2.inRange(hsv, np.array([0, 0, 170]), np.array([180, 110, 255])),
            cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    def reticle(img, base_pale=None):
        """The marker cursor: a bright, DESATURATED, roundish disc.

        GATING IS MANDATORY. Without subtracting the marker-CLOSED pale mask,
        this locks onto static pale scenery at the frame edge (measured: a
        26000px blob at (78,798) present in every stage, including closed).
        Passing base_pale keeps only pixels that became pale when RT opened
        the marker, which correctly isolates the reticle.
        """
        m = pale(img)
        if base_pale is not None:
            m = cv2.bitwise_and(m, cv2.bitwise_not(base_pale))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        best = None
        for c in cnts:
            a = cv2.contourArea(c)
            if a < 700 or a > 40000:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            if bh == 0 or not 0.45 < bw / float(bh) < 2.2:
                continue
            if a / float(bw * bh) < 0.45:
                continue
            mo = cv2.moments(c)
            if mo["m00"] == 0:
                continue
            if best is None or a > best[0]:
                best = (a, mo["m10"] / mo["m00"], mo["m01"] / mo["m00"])
        return best

    ax, ay = stick_xy(aim_dir, args.strength)
    dx, dy = stick_xy(resolve_dir(args.draw), 1.0)

    # Axis names for the chosen sticks. The marker cursor lives on the RIGHT
    # stick; driving lstick moved the reticle only ~9px (measured noise).
    AX, AY = f"{args.aim_stick} x", f"{args.aim_stick} y"
    DX, DY = f"{args.draw_stick} x", f"{args.draw_stick} y"
    # Every stick axis that must be zeroed, regardless of which we drive.
    ZERO = [("lstick x", 0), ("lstick y", 0),
            ("rstick x", 0), ("rstick y", 0)]

    print(f"\n{'=' * 74}")
    what = (f"press X x{args.repeat_x}" if args.mode == "destroy"
            else f"draw={args.draw} for {args.draw_time}s")
    print(f" RUN '{tag}' [{args.mode}]  aim={aim_dir} ({ax:+d},{ay:+d}) "
          f"for {args.aim_time}s   {what}")
    print("=" * 74)

    # -- 0. baseline, marker closed -----------------------------------------
    holder.set_state([])
    send([("r2", 0), ("cross", 0), ("square", 0)] + ZERO)
    time.sleep(1.0)
    print("  [0] baseline (marker closed)")
    closed = grab("0closed")

    # -- 1. hold RT: the marker opens ---------------------------------------
    print(f"  [1] hold RT r2({args.rt}) -> marker opens, time slows")
    holder.set_state([("r2", args.rt)] + ZERO)
    time.sleep(args.settle)
    opened = grab("1open")

    # -- 2. aim the cursor onto the glowing earth ---------------------------
    print(f"  [2] aim {aim_dir}: {AX}({ax}) {AY}({ay}) "
          f"for {args.aim_time}s")
    holder.set_state([("r2", args.rt), (AX, ax), (AY, ay)])
    # Capture MID-AIM, with the deflection STILL applied. Grabbing only after
    # re-centring hid the moving cursor in earlier runs.
    time.sleep(max(0.05, args.aim_time * 0.5))
    midaim = grab("2amidaim", flush=1)
    time.sleep(max(0.05, args.aim_time * 0.5))
    holder.set_state([("r2", args.rt)] + ZERO)
    time.sleep(0.4)
    aimed = grab("2aimed")

    print("      cursor (pale reticle) position by stage:")
    base_pale = pale(closed)          # gate against the marker-CLOSED frame
    pos = {}
    for nm, im in (("open", opened), ("mid-aim", midaim), ("aimed", aimed)):
        r = reticle(im, base_pale)
        pos[nm] = r
        if r is None:
            print(f"        {nm:<8} not found")
        else:
            print(f"        {nm:<8} ({r[1]:7.1f},{r[2]:7.1f})  area={r[0]:.0f}")

    # Did the stick ACTUALLY move the cursor? A wrong stick shows up here as
    # a few px of drift instead of the expected hundreds.
    o, m = pos.get("open"), pos.get("mid-aim")
    if o and m:
        mdx, mdy = m[1] - o[1], m[2] - o[2]
        dist = (mdx * mdx + mdy * mdy) ** 0.5
        print(f"      reticle moved dx={mdx:+.1f} dy={mdy:+.1f} "
              f"dist={dist:.1f}px  on {args.aim_stick}")
        if dist < 25:
            # Only a problem when DESTROYING: erasing needs the cursor to
            # physically reach the pillar. Drawing works despite this tiny
            # measured shift, so do not cry wolf on the proven path.
            if args.mode == "destroy":
                print(f"      !! cursor moved only {dist:.0f}px - too little "
                      f"to reach a pillar.")
                if args.strength < 0.95:
                    print(f"      !! --strength {args.strength} is the likely "
                          f"cause: measured gain is ~450px/s at 1.0 but only "
                          f"~17px/s at 0.4.")
                    print("      !! use --strength 1.0")
                else:
                    print("      !! raise --aim-time (~450px/s, so 0.4s "
                          "~= 180px)")
            else:
                print(f"      (small {dist:.0f}px shift is normal here - "
                      f"drawing still works)")
        else:
            want_x, want_y = DIRECTIONS[aim_dir]
            if (want_x and mdx * want_x < 0) or (want_y and mdy * want_y < 0):
                print(f"      !! moved OPPOSITE to '{aim_dir}' - axis "
                      f"inverted? try the mirrored direction.")

    if args.mode == "destroy":
        # -- 3/4. press X (gimx "square") to erase the aimed-at drawing -----
        # RT MUST stay held the whole time: X only erases while the marker
        # is open. Each press is a discrete tap, not a drag.
        print(f"  [3] press X x{args.repeat_x} "
              f"({args.x_hold}s each) -> erase the drawing")
        anchored = aimed
        # With --hold-aim the stick stays pushed so the cursor cannot drift
        # back to its rest position before X registers.
        aim_axes = ([(AX, ax), (AY, ay)] if args.hold_aim else ZERO)
        if args.hold_aim:
            print(f"      (holding aim {AX}({ax}) {AY}({ay}) during X)")
        for i in range(max(1, args.repeat_x)):
            holder.set_state([("r2", args.rt), ("square", 1)] + aim_axes)
            time.sleep(args.x_hold)
            holder.set_state([("r2", args.rt), ("square", 0)] + aim_axes)
            time.sleep(0.35)
            if i == 0:
                drawing = grab("4destroying", flush=2)
        print("  [5] release RT")
        committed = grab("5committed", flush=2)
    else:
        # -- 3. hold A so the ink anchors to that spot ----------------------
        print(f"  [3] hold A for {args.grab_time}s (anchor the ink)")
        holder.set_state([("r2", args.rt), ("cross", 1)] + ZERO)
        time.sleep(args.grab_time)
        anchored = grab("3anchored", flush=2)

        # -- 4. drag to draw the stroke -------------------------------------
        print(f"  [4] drag {args.draw} on {args.draw_stick} for "
              f"{args.draw_time}s -> pillar grows")
        holder.set_state([("r2", args.rt), ("cross", 1),
                          (DX, dx), (DY, dy)])
        time.sleep(args.draw_time)
        drawing = grab("4drawing", flush=2)

        # -- 5. release A to commit, then RT --------------------------------
        print("  [5] release A (commit), then release RT")
        holder.set_state([("r2", args.rt), ("cross", 0)] + ZERO)
        time.sleep(0.5)
        committed = grab("5committed", flush=2)

    holder.set_state([])
    send([("r2", 0), ("cross", 0), ("square", 0)] + ZERO)
    time.sleep(1.5)
    after = grab("6after")

    print("\n  --- frame deltas (what actually changed) ---")
    print(f"    closed   -> open     : {float(diff(closed, opened)):6.2f}"
          "   (>3 = marker really opened)")
    print(f"    open     -> aimed    : {float(diff(opened, aimed)):6.2f}"
          "   (cursor moved)")
    if args.mode == "destroy":
        print(f"    aimed    -> X-press  : {float(diff(aimed, drawing)):6.2f}"
              "   (>3 = X did something)")
        print(f"    X-press  -> settled  : "
              f"{float(diff(drawing, committed)):6.2f}")
        print(f"    closed   -> after    : {float(diff(closed, after)):6.2f}"
              "   (>3 = world PERMANENTLY changed = drawing ERASED)")
    else:
        print(f"    aimed    -> anchored : "
              f"{float(diff(aimed, anchored)):6.2f}")
        print(f"    anchored -> drawing  : "
              f"{float(diff(anchored, drawing)):6.2f}   (>3 = ink was laid)")
        print(f"    drawing  -> committed: "
              f"{float(diff(drawing, committed)):6.2f}")
        print(f"    closed   -> after    : {float(diff(closed, after)):6.2f}"
              "   (>3 = world PERMANENTLY changed = the pillar stayed)")
    print(f"\n  Frames saved as _mm_{tag}_*.png")


def main() -> None:
    args = parse_args()
    interactive_fill(args)

    aim_dirs = [resolve_dir(a) for a in str(args.aim).split(",") if a.strip()]
    if not aim_dirs:
        raise SystemExit("No aim direction given.")

    load_dotenv_if_present(ROOT / ".env")
    settings = Config.load_all(ROOT / "config", {"settings": "settings.yaml"},
                               base=ROOT)["settings"]

    hw = HardwareBridge(settings)
    holder = Holder()
    try:
        holder.start()
        for i, aim_dir in enumerate(aim_dirs, start=1):
            tag = f"{i}{aim_dir.replace('-', '')}"
            if args.mode == "calib":
                run_calib(hw, holder, args, aim_dir)
            else:
                run_once(hw, holder, args, aim_dir, tag)
            if i < len(aim_dirs):
                try:
                    input("\n  Press Enter for the next aim direction "
                          "(reposition Max first if needed)... ")
                except EOFError:
                    pass
        print("\nDone. If the ink landed in the wrong spot, change --aim "
              "and/or --aim-time and run again.")
        print("Tip: use --strength 0.5 for finer, slower cursor movement.")
    finally:
        # ALWAYS release everything, even on Ctrl-C, or the pad stays stuck
        # holding RT and the game is left in marker mode.
        holder.set_state([])
        holder.stop()
        send([("r2", 0), ("cross", 0), ("square", 0),
              ("lstick x", 0), ("lstick y", 0),
              ("rstick x", 0), ("rstick y", 0)])
        hw.close()


if __name__ == "__main__":
    main()
