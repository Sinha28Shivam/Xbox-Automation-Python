"""combo_dispatch.py - GAME-AGNOSTIC simultaneous multi-control dispatch.

Extracted from the pattern proven twice independently (max_profile.py's
Magic Marker RT+A+stick combo, minecraft_profile.py's move+look+attack) once
it became clear the underlying primitive has nothing to do with either game:
resolving a stick direction / button / trigger name to a real GIMX control,
bundling several into ONE `pad._send_events()` call so they land in a single
controller-state update, holding, then releasing everything together.

WHY THIS IS SEPARATE FROM ANY game_profiles/*.py FILE
-------------------------------------------------------
A profile file owns what its game's moves MEAN (e.g. "sprint" is an LS click
in Minecraft, a stick-held-full-deflection in many other titles). This file
only owns HOW to make several controls happen at once, given their resolved
names from controls.yaml. That split is what makes adding a third game (or a
fourth) a matter of writing a new small catalog of ComboComponent-returning
functions, not re-deriving _send_events plumbing again.

WHY IT MATTERS FOR SPEED
--------------------------
Each gimx.exe subprocess launch costs ~250ms (measured and documented in
test_controller.py's own _send_events docstring). Three controls sent one at
a time is 750ms+ of real wall-clock time before the last one even reaches the
console, and the first one's hold has usually already lapsed by then. Sent
together, it is one ~250ms call - a real reduction in reaction latency, not
just a code-cleanliness win. It's also the only way to express combos that
are physically meaningless if staggered (e.g. jumping while sprint is
already held, not after it).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Literal


@dataclass
class ComboComponent:
    """One control to assert as part of a combo.

    kind="stick"   -> name is a sticks: key (e.g. "left_stick"), direction
                       is a directions: key (e.g. "up"), strength scales the
                       deflection 0..1 (gentler look/aim, full for movement).
    kind="button"  -> name is a buttons: key (e.g. "a", "ls"). Held for the
                       combo's full duration, released with everything else.
    kind="trigger" -> name is a triggers: key (e.g. "rt"). value overrides
                       the configured default_press if a partial pull is
                       ever needed; None uses the full press value.
    """

    kind: Literal["stick", "button", "trigger"]
    name: str
    direction: str | None = None
    strength: float = 1.0
    value: int | None = None


def _stick_events(pad: Any, c: ComboComponent) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    spec = pad.cfg.sticks.get(c.name)
    if spec is None:
        raise KeyError(f"Unknown stick '{c.name}'. Known: {', '.join(pad.cfg.sticks)}")
    dirs = spec.get("directions", {})
    if c.direction not in dirs:
        raise KeyError(f"Unknown direction '{c.direction}' for stick '{c.name}'. "
                       f"Known: {', '.join(dirs)}")
    axis = dirs[c.direction]["axis"]
    magnitude = int(dirs[c.direction]["value"])
    strength = max(0.0, min(1.0, float(c.strength)))
    if strength < 1.0:
        magnitude = int(round(magnitude * strength))
    return [(axis, magnitude)], [(axis, 0)]


def _button_events(pad: Any, c: ComboComponent) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    spec = pad.cfg.buttons.get(c.name)
    if spec is None:
        raise KeyError(f"Unknown button '{c.name}'. Known: {', '.join(pad.cfg.buttons)}")
    control = str(spec["gimx"])
    return [(control, 1)], [(control, 0)]


def _trigger_events(pad: Any, c: ComboComponent) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    spec = pad.cfg.triggers.get(c.name)
    if spec is None:
        raise KeyError(f"Unknown trigger '{c.name}'. Known: {', '.join(pad.cfg.triggers)}")
    control = str(spec["gimx"])
    value = c.value if c.value is not None else int(spec.get("default_press", 32767))
    return [(control, value)], [(control, int(spec.get("min", 0)))]


def resolve_component(pad: Any, c: ComboComponent) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    """One component's (press_events, release_events)."""
    if c.kind == "stick":
        return _stick_events(pad, c)
    if c.kind == "button":
        return _button_events(pad, c)
    if c.kind == "trigger":
        return _trigger_events(pad, c)
    raise ValueError(f"Unknown ComboComponent.kind '{c.kind}'")


def dispatch_combo(pad: Any, components: list[ComboComponent],
                   hold: float, label: str = "combo") -> dict[str, Any]:
    """Assert every component TOGETHER in one GIMX call, hold, release together.

    Returns a small report, never a gameplay verdict - callers decide what a
    successful dispatch actually achieved on screen, same honesty rule as
    every other dispatch function in this framework.
    """
    if not components:
        return {"dispatched": False, "components": 0, "error": "no components"}

    sender = getattr(pad, "_send_events", None)
    if not callable(sender):
        return {"dispatched": False, "components": len(components),
                "error": "pad has no _send_events (unsupported pad implementation)"}

    press_events: list[tuple[str, int]] = []
    release_events: list[tuple[str, int]] = []
    for component in components:
        press, release = resolve_component(pad, component)
        press_events += press
        release_events += release

    sent = bool(sender(press_events, f"{label}:press"))
    time.sleep(max(0.0, float(hold)))
    released = bool(sender(release_events, f"{label}:release"))

    return {"dispatched": sent and released, "components": len(components),
           "hold": hold, "label": label}
