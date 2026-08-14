"""
input_tools.py - the only tools that can change console state.

Everything here delegates to ConsolePad from the existing hardware layer, so
control names, GIMX names, value ranges and timings all come from controls.yaml.
Nothing about the Xbox is encoded here: the same tools drive a PS4 by switching
the console profile, because the mapping lives in config.

THE HONESTY RULE
----------------
Every result carries `dispatched`, never `success`. `dispatched` means GIMX
accepted the event - which, as this project learned the hard way, is compatible
with the console doing absolutely nothing (an unauthenticated session accepts
everything and delivers none of it).

Whether anything actually happened is decided by vision_tools, and only the
verifier gets to say so. Naming the field `dispatched` makes that boundary
impossible to blur by accident: no downstream agent can mistake this for proof.
"""

from __future__ import annotations

import time
from typing import Any

from registry import ToolContext, ToolSpec, fail, make_tool, ok

# The caveat is attached to every single input result. Repetitive by design -
# an agent summarising a run should never be able to read one of these results
# in isolation and conclude the action worked.
_ACK_CAVEAT = (
    "dispatched=true only means GIMX accepted the event. It is NOT evidence "
    "that the console reacted. Capture a frame and compare to confirm.")


def _pad(ctx: ToolContext, console: str | None = None) -> Any:
    return ctx.hardware.pad(console)


# ===========================================================================
# Buttons
# ===========================================================================
def _press_button(ctx: ToolContext) -> Any:
    def run(button: str, times: int = 1, duration: float | None = None,
            interval: float | None = None) -> dict[str, Any]:
        """Press a button one or more times."""
        try:
            pad = _pad(ctx)
        except Exception as exc:
            return fail(f"Controller unavailable: {exc}")

        # Fail fast on an unknown name rather than sending a bad event and
        # decoding GIMX's "Bad button name" from stderr afterwards.
        try:
            kind, canonical = ctx.hardware.controls.resolve(button)
        except KeyError as exc:
            return fail(str(exc), button=button, dispatched=False)

        started = time.time()
        try:
            dispatched = pad.press_times(
                canonical, max(1, int(times)), duration, interval)
        except Exception as exc:
            return fail(f"Error pressing '{button}': {exc}", dispatched=False)

        return ok(
            button=canonical,
            kind=kind,
            times=int(times),
            dispatched=bool(dispatched),
            duration_seconds=round(time.time() - started, 3),
            caveat=_ACK_CAVEAT,
        )

    return make_tool(
        run, "press_button",
        "Press a controller button, optionally N times. Use canonical names "
        "or aliases from get_control_surface (e.g. 'a', 'down', 'guide'). "
        "'interval' adds a pause between repeats - raise it for animated "
        "menus, which can swallow presses that arrive too quickly.")


def _hold_button(ctx: ToolContext) -> Any:
    def run(button: str, seconds: float) -> dict[str, Any]:
        try:
            pad = _pad(ctx)
            _, canonical = ctx.hardware.controls.resolve(button)
        except KeyError as exc:
            return fail(str(exc), dispatched=False)
        except Exception as exc:
            return fail(f"Controller unavailable: {exc}", dispatched=False)

        try:
            dispatched = pad.hold(canonical, float(seconds))
        except Exception as exc:
            return fail(f"Error holding '{button}': {exc}", dispatched=False)

        return ok(button=canonical, seconds=float(seconds),
                  dispatched=bool(dispatched), caveat=_ACK_CAVEAT)

    return make_tool(
        run, "hold_button",
        "Hold a button down for a set number of seconds. Needed for long "
        "presses such as opening the power menu with Guide.")


# ===========================================================================
# Analog
# ===========================================================================
def _move_stick(ctx: ToolContext) -> Any:
    def run(stick: str, direction: str | None = None, x: int | None = None,
            y: int | None = None, duration: float | None = None) -> dict[str, Any]:
        try:
            pad = _pad(ctx)
        except Exception as exc:
            return fail(f"Controller unavailable: {exc}", dispatched=False)

        known = list(ctx.hardware.controls.sticks)
        if stick not in known:
            return fail(f"Unknown stick '{stick}'. Known: {', '.join(known)}",
                        dispatched=False)
        try:
            dispatched = pad.stick(stick, direction, x, y, duration)
        except Exception as exc:
            return fail(f"Error moving stick: {exc}", dispatched=False)

        return ok(stick=stick, direction=direction, x=x, y=y,
                  dispatched=bool(dispatched), caveat=_ACK_CAVEAT)

    return make_tool(
        run, "move_stick",
        "Move an analog stick, either by named direction or by explicit x/y "
        "axis values. The stick returns to centre afterwards.")


def _pull_trigger(ctx: ToolContext) -> Any:
    def run(trigger: str, value: int | None = None,
            duration: float | None = None) -> dict[str, Any]:
        try:
            pad = _pad(ctx)
        except Exception as exc:
            return fail(f"Controller unavailable: {exc}", dispatched=False)
        try:
            dispatched = pad.trigger(trigger, value, duration)
        except Exception as exc:
            return fail(f"Error pulling trigger: {exc}", dispatched=False)
        return ok(trigger=trigger, value=value, dispatched=bool(dispatched),
                  caveat=_ACK_CAVEAT)

    return make_tool(
        run, "pull_trigger",
        "Pull an analog trigger to a value in its configured range "
        "(typically 0-255). Omit the value for a full press.")


# ===========================================================================
# Composites
# ===========================================================================
def _run_macro(ctx: ToolContext) -> Any:
    def run(macro: str) -> dict[str, Any]:
        try:
            pad = _pad(ctx)
        except Exception as exc:
            return fail(f"Controller unavailable: {exc}", dispatched=False)

        known = list(ctx.hardware.controls.macros)
        if macro not in known:
            return fail(f"Unknown macro '{macro}'. Known: {', '.join(known)}",
                        dispatched=False)
        try:
            dispatched = pad.run_macro(macro)
        except Exception as exc:
            return fail(f"Error running macro: {exc}", dispatched=False)
        return ok(macro=macro, dispatched=bool(dispatched), caveat=_ACK_CAVEAT)

    return make_tool(
        run, "run_macro",
        "Run a named macro from controls.yaml - a reusable multi-step "
        "sequence such as returning to the dashboard.")


def _run_special_action(ctx: ToolContext) -> Any:
    def run(action: str) -> dict[str, Any]:
        try:
            pad = _pad(ctx)
        except Exception as exc:
            return fail(f"Controller unavailable: {exc}", dispatched=False)

        spec = ctx.hardware.controls.special.get(action)
        if spec is None:
            known = ", ".join(ctx.hardware.controls.special)
            return fail(f"Unknown action '{action}'. Known: {known}",
                        dispatched=False)

        verified = bool(spec.get("verified", False))
        try:
            dispatched = pad.run_special(action)
        except Exception as exc:
            return fail(f"Error running action: {exc}", dispatched=False)

        return ok(
            action=action,
            dispatched=bool(dispatched),
            # Some sequences in controls.yaml are documented best guesses that
            # were never confirmed on hardware. Surfacing that lets the RCA
            # agent suspect the sequence itself rather than the console.
            hardware_verified=verified,
            warning=None if verified else (
                f"'{action}' is marked verified: false in controls.yaml - the "
                f"sequence is a best guess and may simply be wrong."),
            caveat=_ACK_CAVEAT,
        )

    return make_tool(
        run, "run_special_action",
        "Run a named special action (screenshot, power menu, ...). The result "
        "reports whether that sequence was ever confirmed on real hardware.")


def _run_sequence(ctx: ToolContext) -> Any:
    def run(steps: list[dict[str, Any]]) -> dict[str, Any]:
        try:
            pad = _pad(ctx)
        except Exception as exc:
            return fail(f"Controller unavailable: {exc}", dispatched=False)
        try:
            dispatched = pad.sequence(steps)
        except Exception as exc:
            return fail(f"Error running sequence: {exc}", dispatched=False)
        return ok(steps=len(steps), dispatched=bool(dispatched),
                  caveat=_ACK_CAVEAT)

    return make_tool(
        run, "run_sequence",
        "Run an ad-hoc sequence of steps. Each step is one of "
        "{button, times, interval, duration}, {trigger, value}, "
        "{stick, direction|x|y} or {wait: seconds}.")


# ===========================================================================
# Timing
# ===========================================================================
def _wait(ctx: ToolContext) -> Any:
    def run(seconds: float, reason: str = "") -> dict[str, Any]:
        # Capped so a hallucinated "wait 3600" cannot stall a run. The named
        # timings in controls.yaml (game_launch_wait and friends) are the
        # intended way to wait for something slow.
        limit = ctx.settings.get("runtime.step_timeout_seconds", 60)
        capped = min(float(seconds), float(limit))
        time.sleep(capped)
        return ok(waited_seconds=capped, requested_seconds=float(seconds),
                  capped=capped < float(seconds), reason=reason)

    return make_tool(
        run, "wait",
        "Sleep for a number of seconds. Prefer wait_for_stable_screen when "
        "waiting for a UI transition - a fixed sleep either wastes time or "
        "fires too early.")


def _get_timing(ctx: ToolContext) -> Any:
    def run(key: str = "") -> dict[str, Any]:
        timings = dict(ctx.hardware.controls.timing)
        if key:
            if key not in timings:
                return fail(f"Unknown timing '{key}'. "
                            f"Known: {', '.join(timings)}")
            return ok(key=key, seconds=timings[key])
        return ok(timings=timings)

    return make_tool(
        run, "get_timing",
        "Read the tuned timing values from controls.yaml (menu transitions, "
        "screen loads, game launch). Use these instead of inventing delays.")


# ===========================================================================
# Registration
# ===========================================================================
def provide() -> list[ToolSpec]:
    return [
        ToolSpec("press_button", "Press a button, optionally repeated.",
                 ["input"], _press_button, mutates_hardware=True),
        ToolSpec("hold_button", "Hold a button for N seconds.",
                 ["input"], _hold_button, mutates_hardware=True),
        ToolSpec("move_stick", "Move an analog stick.",
                 ["input"], _move_stick, mutates_hardware=True),
        ToolSpec("pull_trigger", "Pull an analog trigger.",
                 ["input"], _pull_trigger, mutates_hardware=True),
        ToolSpec("run_macro", "Run a named macro from config.",
                 ["input"], _run_macro, mutates_hardware=True),
        ToolSpec("run_special_action", "Run a named special action.",
                 ["input"], _run_special_action, mutates_hardware=True),
        ToolSpec("run_sequence", "Run an ad-hoc step sequence.",
                 ["input"], _run_sequence, mutates_hardware=True),
        ToolSpec("wait", "Sleep for N seconds.", ["timing"], _wait),
        ToolSpec("get_timing", "Read configured timing values.",
                 ["timing", "introspection"], _get_timing),
    ]
