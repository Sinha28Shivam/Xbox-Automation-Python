"""
route_store.py - Persistent route caching for deterministic plan replay.

When a test plan achieves a verified PASS, its exact step sequence is saved
as a known-good route. Subsequent runs of the same scenario can replay that
route directly with 0ms planning latency and zero LLM calls.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from schemas import TestPlan

log = logging.getLogger("route_store")


def _get_route_paths(artifacts_dir: Path | str, scenario_id: str) -> list[Path]:
    """Return prioritized paths to look for a cached route."""
    art_path = Path(artifacts_dir)
    # Handle both run-specific dir (artifacts/runs/run-...) and top-level artifacts
    base_dir = art_path.parent.parent if "runs" in art_path.parts else art_path
    
    candidates = [
        base_dir / "routes" / f"{scenario_id}.route.json",
        Path("config") / "routes" / f"{scenario_id}.route.json",
    ]
    return candidates


def load_cached_route(artifacts_dir: Path | str, scenario_id: str) -> TestPlan | None:
    """Load a verified route if available and valid."""
    for path in _get_route_paths(artifacts_dir, scenario_id):
        if not path.is_file():
            continue
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            plan = TestPlan.model_validate(data)
            if plan.steps:
                log.info(f"Loaded verified cached route for '{scenario_id}' from {path}")
                return plan
        except Exception as exc:
            log.warning(f"Could not load cached route from {path}: {exc}")
    return None


def save_cached_route(artifacts_dir: Path | str, scenario_id: str, plan: TestPlan) -> Path | None:
    """Save a verified plan as a known-good route for repeat replay."""
    if not plan.steps:
        return None
    try:
        art_path = Path(artifacts_dir)
        base_dir = art_path.parent.parent if "runs" in art_path.parts else art_path
        routes_dir = base_dir / "routes"
        routes_dir.mkdir(parents=True, exist_ok=True)
        target = routes_dir / f"{scenario_id}.route.json"
        
        # Save a clean revision-1 copy
        dump = plan.model_dump(mode="json")
        dump["revision"] = 1
        dump["replan_reason"] = None
        with target.open("w", encoding="utf-8") as f:
            json.dump(dump, f, indent=2)
        log.info(f"Saved verified route for '{scenario_id}' to {target}")
        return target
    except Exception as exc:
        log.warning(f"Could not save cached route for '{scenario_id}': {exc}")
        return None


def invalidate_cached_route(artifacts_dir: Path | str, scenario_id: str) -> None:
    """Invalidate a cached route if execution failed."""
    for path in _get_route_paths(artifacts_dir, scenario_id):
        if path.is_file():
            try:
                path.unlink(missing_ok=True)
                log.info(f"Invalidated failed cached route: {path}")
            except Exception as exc:
                log.warning(f"Failed to remove cached route {path}: {exc}")
