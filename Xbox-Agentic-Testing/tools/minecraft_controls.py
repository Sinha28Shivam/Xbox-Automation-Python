"""minecraft_controls.py - load config/minecraft_controls.yaml, the REAL
in-game button mapping read directly from Minecraft Bedrock's own Settings
-> Controller -> Button Mapping screen (see that file's header for the
hardware-verification note). Central lookup so no tool hardcodes a button
literal that could drift from what the game itself actually says.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from config import Config

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_PATH = _PROJECT_ROOT / "config" / "minecraft_controls.yaml"

_cache: Config | None = None


def _load() -> Config:
    global _cache
    if _cache is None:
        _cache = Config.load(_CONFIG_PATH, base=_PROJECT_ROOT)
    return _cache


def button_for(action_key: str) -> str:
    """The real controller button for a named action, e.g. 'attack_destroy'.

    Raises KeyError with the available keys listed if the action is not in
    the config - a loud failure is better than silently falling back to a
    guessed button.
    """
    bindings = _load().section("bindings")
    entry = bindings.get(action_key)
    if not entry:
        raise KeyError(
            f"'{action_key}' is not in minecraft_controls.yaml's bindings. "
            f"Known actions: {', '.join(sorted(bindings))}")
    return str(entry["button"])


def label_for(action_key: str) -> str:
    """The in-game label for a named action, e.g. 'Attack / Destroy'."""
    bindings = _load().section("bindings")
    entry = bindings.get(action_key, {})
    return str(entry.get("label", action_key))


def all_bindings() -> dict[str, Any]:
    return dict(_load().section("bindings"))
