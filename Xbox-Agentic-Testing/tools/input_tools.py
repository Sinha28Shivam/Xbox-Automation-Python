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

from dataclasses import dataclass
import time
from typing import Any

from registry import ToolContext, ToolSpec, fail, make_tool, ok
from vision_tools import read_text_region_impl, warm_ocr_impl

# The caveat is attached to every single input result. Repetitive by design -
# an agent summarising a run should never be able to read one of these results
# in isolation and conclude the action worked.
_ACK_CAVEAT = (
    "dispatched=true only means GIMX accepted the event. It is NOT evidence "
    "that the console reacted. Capture a frame and compare to confirm.")


def _pad(ctx: ToolContext, console: str | None = None) -> Any:
    return ctx.hardware.pad(console)


@dataclass
class KeyboardLayout:
    name: str
    rows: list[list[str]]
    initial_focus: str
    special_keys: dict[str, str]
    search_box_region: dict[str, float]
    typed_text_region: dict[str, float]
    symbol_rows: list[list[str]]
    symbol_toggle_control: str

    def __post_init__(self) -> None:
        self.positions: dict[str, tuple[int, int]] = {}
        for row_index, row in enumerate(self.rows):
            for col_index, label in enumerate(row):
                self.positions[str(label).lower()] = (row_index, col_index)
        self.symbol_positions: dict[str, tuple[int, int]] = {}
        for row_index, row in enumerate(self.symbol_rows):
            for col_index, label in enumerate(row):
                key = str(label).lower()
                if key:
                    self.symbol_positions[key] = (row_index, col_index)

    def supports(self, key: str) -> bool:
        low = key.lower()
        return low in self.positions or low in self.symbol_positions

    def special(self, name: str) -> str | None:
        value = self.special_keys.get(name)
        return str(value).lower() if value else None


def _text_entry_config(ctx: ToolContext) -> dict[str, Any]:
    controls = getattr(ctx.hardware.controls, "data", {}) or {}
    return dict(controls.get("text_entry", {}) or {})


def _text_entry_defaults(ctx: ToolContext) -> dict[str, Any]:
    return dict(_text_entry_config(ctx).get("defaults", {}) or {})


def _keyboard_layouts(ctx: ToolContext) -> dict[str, Any]:
    return dict(_text_entry_config(ctx).get("keyboard_layouts", {}) or {})


def _resolve_keyboard_layout(ctx: ToolContext,
                             keyboard_variant: str | None = None) -> KeyboardLayout:
    defaults = _text_entry_defaults(ctx)
    layouts = _keyboard_layouts(ctx)
    variant = str(keyboard_variant or defaults.get("keyboard_variant", "")).strip()
    if not variant:
        raise ValueError("No keyboard layout is configured for text entry.")
    if variant not in layouts:
        raise ValueError(
            f"Unsupported keyboard_variant '{variant}'. Known: "
            f"{', '.join(sorted(layouts)) or 'none'}")

    spec = dict(layouts[variant] or {})
    rows = [[str(cell).lower() for cell in (row or [])]
            for row in (spec.get("rows") or [])]
    if not rows:
        raise ValueError(f"Keyboard layout '{variant}' has no rows configured.")

    initial = str(spec.get("initial_focus", rows[0][0])).lower()
    specials = {str(k): str(v).lower()
                for k, v in (spec.get("special_keys") or {}).items()}
    region = dict(spec.get("search_box_region") or {})
    typed_region = dict(spec.get("typed_text_region") or {})
    symbol_rows = [[str(cell).lower() for cell in (row or [])]
                   for row in (spec.get("symbol_rows") or [])]
    symbol_toggle = str(spec.get("symbol_toggle_control", "lt")).lower()
    return KeyboardLayout(
        name=variant,
        rows=rows,
        initial_focus=initial,
        special_keys=specials,
        search_box_region=region,
        typed_text_region=typed_region,
        symbol_rows=symbol_rows,
        symbol_toggle_control=symbol_toggle,
    )


def _normalise_text(text: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() or ch == " " else ""
                      for ch in str(text))
    return " ".join(cleaned.split())


def _key_for_char(ch: str, layout: KeyboardLayout) -> str:
    return layout.special("space") if ch == " " else ch.lower()


def _primary_typed_line(text: str) -> str:
    lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    return lines[0] if lines else ""


def _clean_verified_text(observed: str, expected_text: str) -> str:
    """Ignore a trailing caret that OCR often mistakes for a real character."""
    low_observed = str(observed).strip().lower()
    low_expected = str(expected_text).strip().lower()
    if low_expected and low_observed.startswith(low_expected):
        suffix = low_observed[len(low_expected):]
        if suffix in {"l", "i", "1"}:
            return low_expected
    return low_observed


def _common_prefix(a: str, b: str) -> int:
    size = 0
    for left, right in zip(a, b):
        if left != right:
            break
        size += 1
    return size


class TextEntryEngine:
    def __init__(self, ctx: ToolContext, layout: KeyboardLayout,
                 verify_chunk_size: int, max_retries: int):
        self.ctx = ctx
        self.layout = layout
        self.verify_chunk_size = max(1, int(verify_chunk_size))
        self.max_retries = max(0, int(max_retries))
        defaults = _text_entry_defaults(ctx)
        self.move_interval = float(defaults.get("move_interval", 0.3))
        self.key_pause = float(defaults.get("key_pause", 0.2))
        self.recovery_pause = float(defaults.get("recovery_pause", 0.35))
        self.reanchor_each_char = bool(defaults.get("reanchor_each_char", True))
        self.pad = _pad(ctx)
        self.current_key = layout.initial_focus.lower()
        self.current_coord = self.layout.positions.get(self.current_key, (0, 0))

    def type(self, requested_text: str) -> dict[str, Any]:
        requested = str(requested_text)
        expected = _normalise_text(requested)
        if not expected:
            return fail("type_text requires non-empty text.")

        warmed = warm_ocr_impl(self.ctx)
        if not warmed.get("ok"):
            return fail(
                warmed.get("error", "OCR warm-up failed."),
                requested_text=requested,
                keyboard_variant=self.layout.name,
            )

        unsupported = sorted({
            ch for ch in expected
            if not self.layout.supports(_key_for_char(ch, self.layout))
        })
        if unsupported:
            return fail(
                "This keyboard layout cannot type one or more characters.",
                requested_text=requested,
                unsupported_characters=unsupported,
                keyboard_variant=self.layout.name,
            )

        confirmed = ""
        observed = ""
        deferred_mismatch = False

        for index, ch in enumerate(expected):
            dispatch = self._enter_char(ch)
            if not dispatch:
                return fail(
                    "A controller event was not accepted while typing.",
                    requested_text=requested,
                    characters_completed=index,
                    retries_used=0,
                    keyboard_variant=self.layout.name,
                )

            if ((index + 1) % self.verify_chunk_size == 0
                    or index == len(expected) - 1):
                verification = self._verify(expected[:index + 1])
                if not verification.get("ok"):
                    return fail(
                        verification.get("error", "Text verification failed."),
                        requested_text=requested,
                        characters_completed=len(confirmed),
                        typed_text=observed,
                        expected_text=expected[:index + 1],
                        frame_path=verification.get("frame_path"),
                        crop_path=verification.get("crop_path"),
                        retries_used=0,
                        keyboard_variant=self.layout.name,
                    )

                observed = str(verification.get("typed_text", ""))
                if observed == expected[:index + 1]:
                    confirmed = observed
                    continue

                # Mid-word OCR is still noisy on this rig. Record the mismatch,
                # but do not attempt any corrective input. Keep typing and let
                # the final verification decide the verdict.
                if index != len(expected) - 1:
                    deferred_mismatch = True
                    continue

        final_check = self._verify(expected)
        if not final_check.get("ok"):
            return fail(
                final_check.get("error", "Final text verification failed."),
                requested_text=requested,
                expected_text=expected,
                characters_completed=len(confirmed),
                typed_text=observed,
                frame_path=final_check.get("frame_path"),
                crop_path=final_check.get("crop_path"),
                retries_used=0,
                keyboard_variant=self.layout.name,
            )

        observed = str(final_check.get("typed_text", ""))
        if observed != expected:
            return fail(
                "Typed text verification mismatch after typing; no recovery was attempted.",
                requested_text=requested,
                expected_text=expected,
                observed_text=observed,
                characters_completed=_common_prefix(expected, observed),
                frame_path=final_check.get("frame_path"),
                crop_path=final_check.get("crop_path"),
                retries_used=0,
                keyboard_variant=self.layout.name,
                deferred_partial_mismatch=deferred_mismatch,
            )

        return ok(
            requested_text=requested,
            typed_text=observed,
            characters_completed=len(expected),
            retries_used=0,
            verified=True,
            keyboard_variant=self.layout.name,
            dispatched=True,
            deferred_partial_mismatch=deferred_mismatch,
            caveat=_ACK_CAVEAT,
        )

    def _enter_char(self, ch: str) -> bool:
        target = _key_for_char(ch, self.layout)
        if target in self.layout.symbol_positions:
            return self._enter_symbol_char(target)
        if self.reanchor_each_char:
            if not self._reanchor():
                return False
        if not self._navigate_to(target):
            return False
        pressed = self.pad.press("a")
        time.sleep(self.key_pause)
        return bool(pressed)

    def _reanchor(self) -> bool:
        # On the Xbox search keyboard the post-select focus can drift in ways
        # that are hard to model statically. Re-anchoring to the top-left key
        # before each character is slower but much more reliable.
        rows = max(0, len(self.layout.rows) - 1)
        cols = max(0, max(len(row) for row in self.layout.rows) - 1)
        if rows and not self.pad.press_times("up", rows, interval=self.move_interval):
            return False
        if cols and not self.pad.press_times("left", cols, interval=self.move_interval):
            return False
        self.current_key = self.layout.initial_focus.lower()
        self.current_coord = self.layout.positions.get(self.current_key, (0, 0))
        return True

    def _navigate_to(self, target: str) -> bool:
        if target == self.current_key:
            return True
        if target not in self.layout.positions or self.current_key not in self.layout.positions:
            raise ValueError(
                f"Keyboard layout mismatch: cannot navigate from '{self.current_key}' "
                f"to '{target}'.")

        row, col = self.layout.positions[self.current_key]
        target_row, target_col = self.layout.positions[target]
        vertical = target_row - row
        horizontal = target_col - col

        if vertical < 0 and not self.pad.press_times("up", abs(vertical),
                                                     interval=self.move_interval):
            return False
        if vertical > 0 and not self.pad.press_times("down", vertical,
                                                     interval=self.move_interval):
            return False
        if horizontal < 0 and not self.pad.press_times("left", abs(horizontal),
                                                       interval=self.move_interval):
            return False
        if horizontal > 0 and not self.pad.press_times("right", horizontal,
                                                       interval=self.move_interval):
            return False

        self.current_key = target
        self.current_coord = (target_row, target_col)
        return True

    def _enter_symbol_char(self, target: str) -> bool:
        if self.reanchor_each_char:
            if not self._reanchor():
                return False
        if not self.pad.press(self.layout.symbol_toggle_control):
            return False

        target_row, target_col = self.layout.symbol_positions[target]
        row, col = self.current_coord
        vertical = target_row - row
        horizontal = target_col - col

        if vertical < 0 and not self.pad.press_times("up", abs(vertical),
                                                     interval=self.move_interval):
            return False
        if vertical > 0 and not self.pad.press_times("down", vertical,
                                                     interval=self.move_interval):
            return False
        if horizontal < 0 and not self.pad.press_times("left", abs(horizontal),
                                                       interval=self.move_interval):
            return False
        if horizontal > 0 and not self.pad.press_times("right", horizontal,
                                                       interval=self.move_interval):
            return False
        if not self.pad.press("a"):
            return False
        time.sleep(self.key_pause)

        # Automation V3 switched back to the main keyboard after a symbol by
        # pressing LT twice and then continued relative navigation from the
        # same logical coordinate.
        if not self.pad.press(self.layout.symbol_toggle_control):
            return False
        if not self.pad.press(self.layout.symbol_toggle_control):
            return False

        self.current_coord = (target_row, target_col)
        if target_row < len(self.layout.rows) and target_col < len(self.layout.rows[target_row]):
            self.current_key = self.layout.rows[target_row][target_col]
        else:
            self.current_key = self.layout.initial_focus.lower()
        return True

    def _verify(self, expected_text: str) -> dict[str, Any]:
        frame = self.ctx.hardware.capture().grab(allow_blank=True)
        if frame is None:
            return fail("Capture returned no frame while verifying typed text.")

        frame_path = self.ctx.artifacts.save_frame(frame, "text-entry-verify")
        self.ctx.scratch["last_frame_path"] = frame_path
        self.ctx.scratch["last_frame"] = frame
        region = read_text_region_impl(
            self.ctx,
            frame_path=frame_path,
            region=(self.layout.typed_text_region or None),
            preset="" if self.layout.typed_text_region else "search_box",
            keyboard_variant=self.layout.name,
        )
        if not region.get("ok"):
            return region

        observed = _normalise_text(_primary_typed_line(region.get("text", "")))
        observed = _clean_verified_text(observed, expected_text)
        if not observed and expected_text:
            return fail(
                "The search box text could not be read clearly enough to verify typing.",
                frame_path=frame_path,
                crop_path=region.get("crop_path"),
                keyboard_variant=self.layout.name,
            )

        return ok(
            typed_text=observed,
            frame_path=frame_path,
            crop_path=region.get("crop_path"),
            expected_text=expected_text,
            engine=region.get("engine"),
        )

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


def _type_text(ctx: ToolContext) -> Any:
    def run(text: str,
            verify_each_char: bool = False,
            verify_chunk_size: int | None = None,
            max_retries: int | None = None,
            keyboard_variant: str | None = None) -> dict[str, Any]:
        if not ctx.settings.get("verification.ocr.enabled", True):
            return fail(
                "type_text requires OCR verification, but OCR is disabled in settings.")

        try:
            layout = _resolve_keyboard_layout(ctx, keyboard_variant)
        except ValueError as exc:
            return fail(str(exc), requested_text=text, dispatched=False)

        defaults = _text_entry_defaults(ctx)
        chunk = (1 if verify_each_char else
                 int(verify_chunk_size or defaults.get("verify_chunk_size", 2)))
        retries = int(max_retries if max_retries is not None
                      else defaults.get("max_retries", 2))

        try:
            engine = TextEntryEngine(ctx, layout, chunk, retries)
            return engine.type(text)
        except Exception as exc:
            return fail(
                f"type_text failed: {exc}",
                requested_text=text,
                dispatched=False,
                keyboard_variant=layout.name,
            )

    return make_tool(
        run, "type_text",
        "Type text on the Xbox on-screen search keyboard using configured "
        "keyboard geometry plus OCR verification of the search box. Prefer "
        "this over raw directional button spam for search and query entry.")


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
        ToolSpec("type_text", "Type text through the on-screen keyboard.",
                 ["input"], _type_text, mutates_hardware=True),
        ToolSpec("wait", "Sleep for N seconds.", ["timing"], _wait),
        ToolSpec("get_timing", "Read configured timing values.",
                 ["timing", "introspection"], _get_timing),
    ]
