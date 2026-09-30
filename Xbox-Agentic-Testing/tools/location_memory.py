"""location_memory.py - a persistent, coordinate-keyed journal of sightings.

WHY COORDINATE-KEYED, NOT SCREENSHOT-MATCHED
---------------------------------------------
Minecraft has a day/night cycle that changes background brightness and color
substantially. A screenshot of "the village" taken at noon will not visually
match the same village seen again at night, so re-finding a location by
comparing frames would be unreliable. Coordinates from
coordinate_tools.read_player_coordinates are time-invariant, so every entry
here is keyed on (x, y, z) - a saved frame_path is kept only as "what it
looked like when sighted", never as a re-identification signal.

PERSISTENCE ACROSS RUNS
------------------------
Unlike per-run evidence in artifacts/runs/<run_id>/, this journal must
survive across separate scenario runs (an agent exploring today should be
able to recall a village logged yesterday). It lives at the top-level
artifacts/world_memory.json, mirroring how route_store.py persists verified
routes at artifacts/routes/ rather than inside a single run's folder.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from registry import ToolContext, ToolSpec, fail, make_tool, ok
from coordinate_tools import read_player_coordinates_impl


def _memory_path(ctx: ToolContext) -> Path:
    """Cross-run journal file, anchored to the top-level artifacts dir.

    Mirrors route_store.py's own "artifacts/routes/" cross-run pattern:
    ctx.artifacts.run_dir is artifacts/runs/<run_id>/, so its grandparent is
    the shared top-level artifacts/ dir every run writes into.
    """
    run_dir = ctx.artifacts.run_dir
    base_dir = run_dir.parent.parent if "runs" in run_dir.parts else run_dir
    return base_dir / "world_memory.json"


def _load(ctx: ToolContext) -> list[dict[str, Any]]:
    path = _memory_path(ctx)
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _save(ctx: ToolContext, entries: list[dict[str, Any]]) -> None:
    path = _memory_path(ctx)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")


def _distance(a: dict[str, Any], x: float, y: float, z: float) -> float:
    return math.sqrt((a["x"] - x) ** 2 + (a["y"] - y) ** 2 + (a["z"] - z) ** 2)


# ===========================================================================
# Record a sighting
# ===========================================================================
def record_location_impl(
    ctx: ToolContext,
    label: str,
    x: float | None = None,
    y: float | None = None,
    z: float | None = None,
    notes: str = "",
) -> dict[str, Any]:
    """Log a sighting at (x, y, z), or at the player's CURRENT position if
    x/y/z are omitted (reads the live coordinate HUD via coordinate_tools).
    """
    frame_path = None
    if x is None or y is None or z is None:
        coords = read_player_coordinates_impl(ctx)
        if not coords.get("ok"):
            return fail(
                f"No coordinates were given and the live HUD could not be "
                f"read: {coords.get('error')}", label=label)
        x, y, z = coords["x"], coords["y"], coords["z"]
        frame_path = coords.get("frame_path")

    entry = {
        "label": label,
        "x": float(x), "y": float(y), "z": float(z),
        "notes": notes,
        "seen_at": datetime.now(timezone.utc).isoformat(),
        # "As sighted", not a re-identification signal - see module docstring.
        "frame_path": frame_path,
    }
    entries = _load(ctx)
    entries.append(entry)
    _save(ctx, entries)

    return ok(entry=entry, total_entries=len(entries),
             journal_path=str(_memory_path(ctx)))


def _record_location(ctx: ToolContext) -> Any:
    def run(label: str, x: float | None = None, y: float | None = None,
            z: float | None = None, notes: str = "") -> dict[str, Any]:
        return record_location_impl(ctx, label=label, x=x, y=y, z=z, notes=notes)

    return make_tool(
        run, "record_location",
        "Log a labeled sighting (e.g. 'village', 'tree cluster') at a given "
        "(x, y, z), or at the player's CURRENT position if x/y/z are "
        "omitted - reads the live coordinate HUD. Persists across runs so "
        "a location found in one session can be recalled in another.")


# ===========================================================================
# Recall sightings
# ===========================================================================
def find_nearest_locations_impl(
    ctx: ToolContext,
    x: float | None = None,
    y: float | None = None,
    z: float | None = None,
    label_contains: str = "",
    max_results: int = 5,
) -> dict[str, Any]:
    """Nearest logged sightings to (x, y, z), or to the player's CURRENT
    position if x/y/z are omitted. Optionally filter by label substring.
    """
    if x is None or y is None or z is None:
        coords = read_player_coordinates_impl(ctx)
        if not coords.get("ok"):
            return fail(
                f"No coordinates were given and the live HUD could not be "
                f"read: {coords.get('error')}")
        x, y, z = coords["x"], coords["y"], coords["z"]

    entries = _load(ctx)
    if label_contains:
        entries = [e for e in entries
                  if label_contains.lower() in str(e.get("label", "")).lower()]

    ranked = sorted(entries, key=lambda e: _distance(e, x, y, z))
    top = ranked[:max(1, int(max_results))]
    for e in top:
        e["distance"] = round(_distance(e, x, y, z), 2)

    return ok(from_x=x, from_y=y, from_z=z,
             results=top, total_matching=len(entries),
             journal_path=str(_memory_path(ctx)))


def _find_nearest_locations(ctx: ToolContext) -> Any:
    def run(x: float | None = None, y: float | None = None,
            z: float | None = None, label_contains: str = "",
            max_results: int = 5) -> dict[str, Any]:
        return find_nearest_locations_impl(
            ctx, x=x, y=y, z=z, label_contains=label_contains,
            max_results=max_results)

    return make_tool(
        run, "find_nearest_locations",
        "Recall the nearest logged sightings to a given (x, y, z), or to "
        "the player's CURRENT position if x/y/z are omitted. Optionally "
        "filter by a label substring (e.g. 'village'). Returns each match's "
        "distance, sorted nearest first.")


def provide() -> list[ToolSpec]:
    return [
        ToolSpec(name="record_location",
                 description="Log a labeled sighting at a coordinate for later recall.",
                 tags=["analysis", "memory"],
                 factory=_record_location, mutates_hardware=False),
        ToolSpec(name="find_nearest_locations",
                 description="Recall the nearest logged sightings to a coordinate.",
                 tags=["analysis", "memory"],
                 factory=_find_nearest_locations, mutates_hardware=False),
    ]
