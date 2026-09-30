"""Deterministic, measurement-verified chest looting for the Minecraft chest screen.

Instead of asking a vision model where the cursor is, this reads the frame:
  * cursor cell  - the pointer glyph makes ~1374 near-white pixels in its cell
                   (baseline 500-950)
  * occupancy    - empty slots are flat gray; items deviate from the modal value

Geometry is for a 1920x1080 capture; single (3 row) and double (6 row) chests
are auto-selected from LAYOUTS. A transfer is counted only when the chest slot
Y was pressed on is measured EMPTY afterwards.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np

CELL = 62
PITCH = 72
X0 = 640
LAYOUTS = (  # (chest rows y, inventory rows y): double chest first, then single
    (tuple(188 + 72 * n for n in range(6)), (664, 736, 808, 892)),
    ((296, 368, 440), (556, 628, 700, 784)),
)
COLS = 9
CURSOR_MIN = 1200
OCC_MIN = 300

Cell = tuple[int, int]


def _cursor_score(a: np.ndarray, y: int, x: int) -> int:
    return int((a[y - 6:y + 68, x - 6:x + 68].min(axis=2) >= 245).sum())


def _occ_score(a: np.ndarray, y: int, x: int) -> int:
    inner = a[y + 4:y + 58, x + 4:x + 58].mean(axis=2).astype(np.int32)
    modal = int(np.bincount(inner.ravel()).argmax())
    return int((np.abs(inner - modal) > 25).sum())


def _serpentine(rows: int) -> list[Cell]:
    order: list[Cell] = []
    for r in range(rows):
        cols = range(COLS) if r % 2 == 0 else range(COLS - 1, -1, -1)
        order += [(r, c) for c in cols]
    return order


def loot_chest_grid(ctx: Any, max_steps: int = 220) -> dict[str, Any]:
    from gameplay_engine import grab_nonblank

    pad = ctx.hardware.pad()
    cam = ctx.hardware.capture()
    occ: dict[Cell, bool | None] = {}
    lay: dict[str, Any] = {"chest": LAYOUTS[0][0], "inv": LAYOUTS[0][1], "fixed": False}
    res: dict[str, Any] = {
        "macro": "loot_chest", "dispatched": False, "initial_occupied": 0,
        "verified_transfers": 0, "failed_transfers": 0, "remaining_occupied": 0,
        "inv_items_start": None, "inv_items_end": None, "steps": 0, "error": None,
    }
    state: dict[str, Any] = {"frame": None, "cursor": None}

    def observe() -> Cell | None:
        frame, _ = grab_nonblank(cam)
        if frame is None:
            state["cursor"] = None
            return None
        a = np.ascontiguousarray(frame[:, :, ::-1])
        state["frame"] = frame
        best, best_score = None, CURSOR_MIN - 1
        cands = [(lay["chest"], lay["inv"])] if lay["fixed"] else LAYOUTS
        for chest_y, inv_y in cands:
            for ri, y in enumerate(chest_y + inv_y):
                for c in range(COLS):
                    s = _cursor_score(a, y, X0 + PITCH * c)
                    if s > best_score:
                        best, best_score = (ri, c), s
                        lay["chest"], lay["inv"] = chest_y, inv_y
        if best is None:
            state["cursor"] = None
            return None
        lay["fixed"] = True
        chest_y, inv_y = lay["chest"], lay["inv"]
        n = len(chest_y)
        for ri, y in enumerate(chest_y):
            for c in range(COLS):
                if (ri, c) != best:
                    occ[(ri, c)] = _occ_score(a, y, X0 + PITCH * c) >= OCC_MIN
                else:
                    occ.setdefault((ri, c), None)
        state["inv"] = sum(_occ_score(a, y, X0 + PITCH * c) >= OCC_MIN
                           for i, y in enumerate(inv_y) for c in range(COLS) if (n + i, c) != best)
        state["cursor"] = best
        return best

    def step(direction: str) -> None:
        res["steps"] += 1
        pad.press(direction, duration=0.15)
        time.sleep(0.35)

    def go_to(target: Cell, tries: int = 12) -> bool:
        for _ in range(tries):
            cur = observe()
            if cur is None:
                time.sleep(0.3)
                continue
            if cur == target:
                return True
            if cur[0] > target[0]:
                step("up")
            elif cur[0] < target[0]:
                step("down")
            elif cur[1] > target[1]:
                step("left")
            else:
                step("right")
        return False

    if observe() is None:
        # Cursor is not drawn until the pad is used: nudge and re-observe.
        for nudge in ("right", "down", "left", "up", "right"):
            step(nudge)
            if observe() is not None:
                break
    if state["cursor"] is None:
        if ctx.artifacts is not None and state["frame"] is not None:
            ctx.artifacts.save_frame(state["frame"], "chestloot-fail")
        res["error"] = "cursor not found on chest screen"
        return res
    res["dispatched"] = True
    if ctx.artifacts is not None and state["frame"] is not None:
        ctx.artifacts.save_frame(state["frame"], "chestloot-start")

    if not go_to((0, 0)):
        res["error"] = "could not reach chest slot (0,0)"
        return res
    # Cursor hides its own cell: peek at (0,0) from the neighbour, then return.
    step("right")
    observe()
    res["initial_occupied"] = sum(1 for v in occ.values() if v)
    res["inv_items_start"] = state.get("inv")
    if not go_to((0, 0)):
        res["error"] = "could not return to chest slot (0,0)"
        return res

    pending: Cell | None = None
    for target in _serpentine(len(lay['chest'])):
        if res["steps"] >= max_steps:
            break
        if not go_to(target):
            res["error"] = f"lost cursor navigating to {target}"
            break
        if pending is not None:
            if occ.get(pending) is False:
                res["verified_transfers"] += 1
            else:
                res["failed_transfers"] += 1
            pending = None
        if occ.get(target):
            pad.press("y", duration=0.15)
            time.sleep(0.5)
            pending = target

    if pending is not None:
        r, c = pending
        step("left" if c > 0 else "right")
        observe()
        if occ.get(pending) is False:
            res["verified_transfers"] += 1
        else:
            res["failed_transfers"] += 1

    observe()
    res["remaining_occupied"] = sum(1 for v in occ.values() if v)
    res["inv_items_end"] = state.get("inv")
    if ctx.artifacts is not None and state["frame"] is not None:
        ctx.artifacts.save_frame(state["frame"], "chestloot-end")
    return res
