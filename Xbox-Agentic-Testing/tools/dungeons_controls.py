"""Loader for config/dungeons_controls.yaml (Minecraft Dungeons bindings)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from config import Config

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_PATH = _PROJECT_ROOT / "config" / "dungeons_controls.yaml"

_cache: Config | None = None


def _load() -> Config:
    global _cache
    if _cache is None:
        _cache = Config.load(_CONFIG_PATH, base=_PROJECT_ROOT)
    return _cache


def button_for(action_key: str) -> str:
    bindings = _load().section("bindings")
    entry = bindings.get(action_key)
    if not entry:
        raise KeyError(f"'{action_key}' is not in dungeons_controls.yaml. "
                       f"Known: {', '.join(sorted(bindings))}")
    return str(entry["button"])


def all_bindings() -> dict[str, Any]:
    return dict(_load().section("bindings"))
