"""
menu_navigator.py - vision-guided menu navigation.

WHY THIS EXISTS
---------------
Launching a game used to be a KEYWORD CLASSIFIER over OCR text, and every bug
it produced was a keyword bug:

  * the word "continue" was treated as a title-screen prompt, so the launcher
    pressed A on Max's pre-focused CONTINUE item and resumed the LAST SAVE -
    landing in "Chapter 4-2: The Great Fall" instead of the requested level;
  * "chapter" and "select level" appear on BOTH the main menu and the level
    picker, so the two screens could not be told apart;
  * how far SELECT LEVEL sat below CONTINUE was a HARDCODED offset, which is
    a guess about menu layout that breaks the moment the layout changes.

Meanwhile the vision model was already reading those screens perfectly - it
reported "CONTINUE (highlighted in orange), SELECT LEVEL, OPTIONS, HELP,
ACHIEVEMENTS" - but had no menu macros to act with, so the best it could do
was press A. The intelligence was there; the action vocabulary was not.

So this module gives the model the vocabulary: read the screen, say WHICH item
carries the highlight, and move the focus one row at a time until the wanted
item is focused. No offsets, no keyword lists.

WHY NOT detect_focus_highlight
------------------------------
That tool hunts for GREEN, with an HSV range narrowed to the Xbox dashboard's
focus ring. Max highlights its menu in ORANGE, so it cannot see focus here at
all. A model reading the highlight is the only approach that generalises
across a dashboard, a game's own menu and a level picker.

THE HONESTY RULE
----------------
`press_a` is gated IN CODE, not by asking the prompt nicely: the focused item
must actually resemble the target before a confirm is dispatched. A model can
hallucinate focus, and confirming the wrong menu item is destructive (it can
resume the wrong save or restart a level). When the check fails we spend a
cycle moving instead - a wasted d-pad press is always cheaper than a wrong
confirm.
"""
from __future__ import annotations

import base64
import difflib
import re
import threading
import time
from typing import Any, Literal

import cv2
import numpy as np
from pydantic import BaseModel, Field


# ===========================================================================
# What the model is allowed to decide
# ===========================================================================
class MenuDecision(BaseModel):
    """The model's reading of one menu frame, plus the single next input."""

    screen: Literal[
        "dashboard",
        "game_tile_focused",
        "publisher_logo",
        "title_screen",
        "main_menu",
        "level_picker",
        "chapter_list",
        "save_slot_picker",
        "in_gameplay",
        "loading",
        "store_or_purchase",
        "error_dialog",
        "unknown",
    ] = Field(description="Which kind of screen is on display right now.")

    visible_items: list[str] = Field(
        default_factory=list,
        description="Menu items/tiles you can read, TOP TO BOTTOM (or left to "
                    "right for a horizontal row). Empty for a logo, a loading "
                    "screen or gameplay.")

    focused_item: str = Field(
        default="",
        description="The ONE item that currently carries the selection "
                    "highlight. Empty string if nothing is clearly focused - "
                    "say empty rather than guessing.")

    focus_evidence: str = Field(
        default="",
        description="The visual cue proving that focus, e.g. 'CONTINUE is "
                    "filled orange while the others are grey', 'the tile has "
                    "a white outline and is scaled up'. Required whenever "
                    "focused_item is set.")

    target_visible: bool = Field(
        default=False,
        description="Is the item we are trying to reach readable on screen?")

    target_item: str = Field(
        default="",
        description="The on-screen item that best matches the goal, spelled "
                    "exactly as it appears. Empty if it is not visible.")

    action: Literal[
        "press_a",
        "press_b",
        "dpad_up",
        "dpad_down",
        "dpad_left",
        "dpad_right",
        "bumper_left",
        "bumper_right",
        "wait",
        "done",
        "abort",
    ] = Field(description="The ONE input to send now, or done/abort.")

    repeat: int = Field(
        default=1, ge=1, le=10,
        description="How many times to send that input. Use it to cross "
                    "several menu rows at once, e.g. dpad_down repeat=3. "
                    "Keep it at 1 when you are unsure.")

    reasoning: str = Field(
        default="",
        description="Step by step: what is focused now, what needs focus, and "
                    "why this input closes the gap.")

    confidence: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Confidence that this input moves us toward the goal.")


# ===========================================================================
# System prompt
# ===========================================================================
MENU_SYSTEM_PROMPT = """\
You are driving an Xbox One through its MENUS with an emulated controller. The
attached image is the LIVE screen right now. Reason only from what you can
actually see in it.

YOUR JOB
  Reach the goal stated below by moving the SELECTION HIGHLIGHT onto the right
  item and then confirming it. One input per turn.

INPUTS AVAILABLE
  dpad_up / dpad_down       move the highlight one row
  dpad_left / dpad_right    move the highlight one column
  bumper_left/bumper_right  switch tab/chapter/page (LB / RB)
  press_a                   CONFIRM the currently focused item
  press_b                   go back / cancel
  wait                      the screen is loading or animating; do nothing
  done                      the goal is reached - gameplay has started
  abort                     the goal is impossible from here (see below)

THE ONE RULE THAT MATTERS: NEVER CONFIRM AN UNWANTED ITEM
  `press_a` confirms WHATEVER IS FOCUSED, not what you wish were focused. So:
    1. Read `focused_item` off the screen - which item is highlighted?
    2. Compare it to the item the goal needs.
    3. If they are NOT the same, return a d-pad move, NOT press_a.
  Only return press_a when the item you want IS ALREADY the focused one.

  This is not hypothetical. On this exact game the launcher pressed A while
  CONTINUE was focused; that resumed a save in a completely different chapter
  and the whole run was wasted.

GETTING TO A LEVEL IS A JOURNEY, NOT ONE CLICK
  The item you confirm on each screen is usually NOT the final goal - it is
  the next step toward it:
      game tile      -> confirm PLAY
      title screen   -> press A to advance
      main menu      -> confirm SELECT LEVEL   (never CONTINUE)
      chapter list   -> confirm the CHAPTER that holds your level
      level list     -> confirm the LEVEL itself
  So on the main menu the right answer is `press_a` with SELECT LEVEL focused,
  even though SELECT LEVEL is not the level you want. Do not refuse to confirm
  a waypoint just because it is not the destination, and do not hunt for the
  level's name on a screen that does not list levels yet.

A SCREEN THAT DOES NOT CHANGE MEANS YOUR INPUT DID NOTHING
  Check the history below. If the last few cycles show the same screen and the
  highlight is just moving up and down, stop moving and CONFIRM the waypoint
  that leads onward. Splash screens and logos ignore the d-pad entirely - they
  only respond to A.

CONTINUE / RESUME IS A TRAP
  A game's main menu usually pre-focuses CONTINUE or RESUME, because that is
  what a human wants most of the time. It resumes THE LAST SAVE, which is NOT
  a specific requested level. When the goal names a level, you must go to
  SELECT LEVEL / LEVEL SELECT / CHAPTERS instead - move the highlight DOWN off
  CONTINUE first, then confirm.

DESTRUCTIVE ITEMS - NEVER CONFIRM THESE
  "Restart Level", "Restart from Checkpoint", "New Game", "Delete Save",
  "Return to Main Menu", or any "you will lose your progress" warning. They
  throw away progress. Return press_b to back out instead.

WHEN TO ABORT
  * screen is `store_or_purchase` - the account cannot launch this game
  * screen is `error_dialog` and B does not clear it
  Abort is honest and useful. Pressing A hoping for the best is not.

READING FOCUS HONESTLY
  Focus looks like: a filled/coloured row against grey siblings, a bright
  outline, a scaled-up tile, or a moving caret. Set `focus_evidence` to the
  cue you actually saw. If NOTHING is clearly highlighted, leave
  `focused_item` empty and use a d-pad nudge to make the highlight visible -
  do not invent one, and do not press A to "find out".

COUNTING ROWS
  `visible_items` is ordered as drawn. To go from item i to item j, that is
  (j - i) presses of dpad_down when j is below i. Set `repeat` accordingly,
  but prefer repeat=1 when the list is long or partly cut off - you get a
  fresh screenshot after every turn, so small steps are cheap and safe.

LOGOS, TITLE CARDS AND LOADING
  * publisher_logo / title_screen -> press_a to advance past it
  * loading -> wait
  * in_gameplay -> done (the level has started; there is nothing left to click)
"""


# ===========================================================================
# Navigation
# ===========================================================================
class MenuNavigator:
    """Vision-guided menu walker. Owns no hardware; borrows the bridge."""

    # Config-driven; see settings.yaml gameplay.menu.*
    DEFAULTS = {
        "max_cycles": 25,
        "settle_seconds": 1.5,
        "llm_timeout_seconds": 90.0,
        "press_duration": 0.2,
        # A CONFIRM changes screen, and that transition is slower than a
        # highlight move: a menu slides out, a submenu animates in. Rather
        # than guess a fixed delay we WAIT FOR THE PIXELS TO CHANGE, up to
        # this long, so the next decision is never made on the old screen.
        "confirm_timeout": 8.0,
        "change_poll": 0.3,
        # Mean per-pixel delta that counts as "the screen actually changed".
        # Menu transitions move a large fraction of the frame, so this can sit
        # well above capture noise.
        "change_delta": 6.0,
        # A HIGHLIGHT MOVE is not a screen transition: only one or two menu
        # rows repaint, so MEAN delta is the wrong metric - a row is well
        # under 1% of a 1080p frame, which averages away to roughly the same
        # number as compression noise. Instead we measure the FRACTION OF
        # PIXELS that changed strongly, which is what a moving highlight
        # actually looks like: a small area changing a lot, rather than the
        # whole frame changing a little.
        "move_pixel_fraction": 0.0015,   # 0.15% of the frame
        "move_pixel_delta": 30,          # per-pixel change that counts
        # A highlight move is nearly instant; it does not need the confirm's
        # patience, and a shorter wait keeps the cycle budget usable.
        "move_timeout": 2.5,
    }

    # Words that must never be confirmed, whatever the model says.
    DESTRUCTIVE = (
        "restart", "new game", "delete", "erase", "return to main",
        "quit", "exit game", "lose your progress", "overwrite",
    )

    # Items that resume the LAST SAVE rather than starting a chosen level.
    # Confirming one of these is the original bug: it landed the run in
    # "Chapter 4-2: The Great Fall" instead of Sea of Sand.
    RESUME_ITEMS = ("continue", "resume", "last checkpoint")

    # Items that are a STEP TOWARD a level rather than the level itself.
    # Confirming these is how you ever reach the level list at all, so they
    # are allowed even though they do not match the final target.
    WAYPOINTS = (
        "play", "start", "select level", "level select", "levels",
        "chapter", "chapters", "story", "campaign", "single player",
        "press a",
    )

    def __init__(self, hardware: Any, settings: Any, artifacts: Any = None,
                 llm_factory: Any = None):
        self.hardware = hardware
        self.settings = settings
        self.artifacts = artifacts
        self._llm_factory = llm_factory
        self._runnable: Any | None = None
        # Populated by navigate_to; read by the mechanics report so the
        # launch journey can be shown alongside the gameplay evidence.
        self.steps: list[dict[str, Any]] = []
        # Set once a confirm actually lands on the requested target, so a
        # "gameplay reached" result can say whether the RIGHT level started
        # rather than just that some level did.
        self._confirmed_target = False

    # -- config ------------------------------------------------------------
    def _cfg(self, key: str) -> Any:
        return self.settings.get(f"gameplay.menu.{key}", self.DEFAULTS[key])

    # -- model -------------------------------------------------------------
    def _build_runnable(self) -> Any:
        """A structured, vision-capable runnable returning MenuDecision."""
        from llm import LLMFactory, structured

        factory = self._llm_factory or LLMFactory(self.settings)
        provider = factory.default_provider
        if not factory.supports_vision(provider):
            raise RuntimeError(
                f"LLM provider '{provider}' is not marked supports_vision in "
                f"settings.yaml, so it cannot look at menu screens. Vision-"
                f"guided launch needs a multimodal model (anthropic / openai "
                f"/ google).")
        return structured(factory.build(provider=provider), MenuDecision)

    @property
    def runnable(self) -> Any:
        if self._runnable is None:
            self._runnable = self._build_runnable()
        return self._runnable

    def _invoke_with_timeout(self, messages: Any, timeout: float) -> Any:
        """Run the model on a daemon thread so a stuck call cannot hang us."""
        box: dict[str, Any] = {}

        def work() -> None:
            try:
                box["value"] = self.runnable.invoke(messages)
            except Exception as exc:
                box["error"] = exc

        worker = threading.Thread(target=work, daemon=True)
        worker.start()
        worker.join(max(5.0, float(timeout)))
        if worker.is_alive():
            raise TimeoutError(f"vision model exceeded {timeout:.0f}s")
        if "error" in box:
            raise box["error"]
        return box["value"]

    # -- perception --------------------------------------------------------
    @staticmethod
    def _encode(frame: np.ndarray) -> str:
        h, w = frame.shape[:2]
        if w > 1280:
            scale = 1280 / w
            frame = cv2.resize(frame, (1280, int(h * scale)),
                               interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
        return base64.b64encode(buf).decode("utf-8") if ok else ""

    # -- the confirm gate --------------------------------------------------
    @staticmethod
    def _norm(text: str) -> str:
        return " ".join(re.sub(r"[^a-z0-9 ]+", " ", str(text).lower()).split())

    @classmethod
    def _matches(cls, focused: str, target: str) -> bool:
        """Is the focused item really the target? Fuzzy, but not generous.

        The model reads the labels, so what we guard against is it naming an
        item slightly differently between two fields - not character noise.
        Substring or a high ratio both count; anything looser would defeat
        the point of the gate.
        """
        f, t = cls._norm(focused), cls._norm(target)
        if not f or not t:
            return False
        if f == t or f in t or t in f:
            return True
        return difflib.SequenceMatcher(None, f, t).ratio() >= 0.8

    @classmethod
    def _is_destructive(cls, label: str) -> bool:
        low = cls._norm(label)
        return any(word in low for word in cls.DESTRUCTIVE)

    @classmethod
    def _is_resume(cls, label: str) -> bool:
        """Would confirming this resume the last save instead of our level?

        Kept separate from the waypoint list below: CONTINUE is the item the
        original bug confirmed, and it must stay vetoed even though it sits on
        the same menu as the legitimate waypoints.
        """
        low = cls._norm(label)
        return any(word in low for word in cls.RESUME_ITEMS)

    @classmethod
    def _is_waypoint(cls, label: str) -> bool:
        """Is this an item that leads TOWARD a level, rather than being one?

        Reaching a level is a journey: PLAY -> SELECT LEVEL -> CHAPTER 1 ->
        the level. Only the last hop equals the target, so the intermediate
        hops need their own permission or the gate deadlocks on the very item
        that makes progress possible.
        """
        low = cls._norm(label)
        if any(word in low for word in cls.RESUME_ITEMS):
            return False           # CONTINUE is never a waypoint
        return any(word in low for word in cls.WAYPOINTS)

    # -- action ------------------------------------------------------------
    def _dispatch(self, action: str, repeat: int) -> bool:
        """Send one menu input. Returns whether anything was dispatched."""
        pad = self.hardware.pad()
        press = float(self._cfg("press_duration"))
        button = {
            "press_a": "a",
            "press_b": "b",
            "dpad_up": "up",
            "dpad_down": "down",
            "dpad_left": "left",
            "dpad_right": "right",
            "bumper_left": "lb",
            "bumper_right": "rb",
        }.get(action)

        if button is None:
            return False

        sent = False
        for _ in range(max(1, min(int(repeat), 10))):
            sent = bool(pad.press(button, duration=press)) or sent
            # Menus animate. Without a gap between presses the console
            # coalesces them and the highlight moves one row instead of three.
            time.sleep(0.35)
        return sent

    # -- settling ----------------------------------------------------------
    def _wait_for_change(self, before: np.ndarray | None,
                         timeout: float,
                         threshold: float | None = None) -> tuple[bool, float]:
        """Block until the screen differs from `before`, or `timeout` passes.

        WHY THIS EXISTS
        A confirm is not finished when the button is released. Observed live:
        after A was pressed on SELECT LEVEL, the very next screenshot still
        showed the main menu with SELECT LEVEL focused, so the model - quite
        reasonably - confirmed it a SECOND time. That second A landed on the
        level list and launched whatever was focused there (Prologue), while
        the loop still believed it was on the main menu.

        A fixed sleep cannot fix this: menu transitions vary with load time.
        Polling the pixels does, and it also returns as soon as the screen is
        ready rather than always paying the worst case.
        """
        if before is None:
            time.sleep(float(self._cfg("settle_seconds")))
            return False, 0.0

        poll = float(self._cfg("change_poll"))
        if threshold is None:
            threshold = float(self._cfg("change_delta"))
        deadline = time.time() + max(0.5, float(timeout))
        best = 0.0

        try:
            camera = self.hardware.capture()
        except Exception:
            time.sleep(float(self._cfg("settle_seconds")))
            return False, 0.0

        while time.time() < deadline:
            time.sleep(poll)
            try:
                now = camera.grab(allow_blank=True)
            except Exception:
                continue
            if now is None or now.shape != before.shape:
                continue
            delta = float(np.mean(cv2.absdiff(now, before)))
            best = max(best, delta)
            if delta >= threshold:
                # Changed. Give the animation a moment to finish so the next
                # screenshot is a settled screen, not a mid-slide blur.
                time.sleep(float(self._cfg("settle_seconds")))
                return True, delta
        return False, best

    def _wait_for_move(self, before: np.ndarray | None,
                       timeout: float) -> tuple[bool, float]:
        """Block until the HIGHLIGHT visibly moves, or `timeout` passes.

        Separate from _wait_for_change because the signals are different
        shapes. A screen transition changes most of the frame a little; a
        highlight move changes a SMALL AREA a lot. Averaging the second one
        over a 1080p frame buries it in compression noise, so here we count
        the fraction of strongly-changed pixels instead.

        Returns (moved, fraction_changed).
        """
        if before is None:
            time.sleep(float(self._cfg("settle_seconds")))
            return False, 0.0

        poll = float(self._cfg("change_poll"))
        min_fraction = float(self._cfg("move_pixel_fraction"))
        pixel_delta = int(self._cfg("move_pixel_delta"))
        deadline = time.time() + max(0.5, float(timeout))
        best = 0.0

        try:
            camera = self.hardware.capture()
        except Exception:
            time.sleep(float(self._cfg("settle_seconds")))
            return False, 0.0

        while time.time() < deadline:
            time.sleep(poll)
            try:
                now = camera.grab(allow_blank=True)
            except Exception:
                continue
            if now is None or now.shape != before.shape:
                continue
            diff = cv2.absdiff(now, before)
            if diff.ndim == 3:
                diff = diff.max(axis=2)
            fraction = float(np.count_nonzero(diff >= pixel_delta)) / diff.size
            best = max(best, fraction)
            if fraction >= min_fraction:
                # Let the highlight animation finish before the next read.
                time.sleep(0.25)
                return True, fraction
        return False, best

    # -- the gate ----------------------------------------------------------
    def _gate_confirm(self, d: MenuDecision, goal: str) -> tuple[str, int, str]:
        """Veto a press_a that would confirm the wrong or a dangerous item.

        Returns the action to actually run, its repeat count, and a note. This
        is deliberately CODE, not prompt guidance: the prompt already forbids
        confirming an unfocused item, and the prompt was not enough - a model
        that has convinced itself SELECT LEVEL is focused will happily say
        press_a. Only what we can check ourselves is a real safeguard.
        """
        if d.action != "press_a":
            return d.action, d.repeat, ""

        # 1. Never confirm something that destroys progress. Checked FIRST,
        #    because this is true on any screen.
        if self._is_destructive(d.focused_item):
            return ("press_b", 1,
                    f"VETO press_a: '{d.focused_item}' looks destructive.")

        # 2. Screens that are not item lists: A just means "advance". There is
        #    no menu row to match, and a logo legitimately reports NO focused
        #    item - so this must be decided BEFORE the no-focus rule below.
        #    Getting that order wrong made the launcher answer four logo/title
        #    screens in a row with dpad_down, which does nothing to a splash
        #    screen and burned cycles 5-8 of the budget.
        if d.screen in ("publisher_logo", "title_screen", "loading",
                        "game_tile_focused", "dashboard"):
            return d.action, d.repeat, ""

        # 3. On a real menu, nothing readable focused - move to reveal it.
        if not d.focused_item.strip():
            return ("dpad_down", 1,
                    "VETO press_a: no focused item was reported on a menu, so "
                    "a confirm would hit whatever happens to be selected.")

        # 4. CONTINUE / RESUME resumes the wrong save whenever a specific
        #    level was asked for. This is the original bug and is never
        #    allowed through while we have a target.
        if self._is_resume(d.focused_item) and goal.strip():
            return ("dpad_down", 1,
                    f"VETO press_a: '{d.focused_item}' would resume the last "
                    f"save instead of starting '{goal.strip()}'.")

        # 5. The target itself - the thing we came here to confirm.
        #
        #    The CALLER's target wins over `d.target_item`. Trusting the
        #    model's own field would let it nominate a goal and then approve
        #    itself: reporting target_item="Black Rock Canyon" while focus
        #    sits on Black Rock Canyon matches trivially and would confirm
        #    the WRONG LEVEL.
        target = goal.strip() or d.target_item.strip()
        if self._matches(d.focused_item, target):
            return d.action, d.repeat, ""

        # 6. A NAVIGATION WAYPOINT. Reaching a level is a multi-step journey -
        #    PLAY, then SELECT LEVEL, then CHAPTER 1, and only then the level.
        #    Requiring focus to equal the FINAL target at every step made the
        #    gate veto SELECT LEVEL itself, so the launcher could never leave
        #    the main menu: it moved the highlight up and down for 13 cycles
        #    while refusing the one item that led onward.
        if self._is_waypoint(d.focused_item):
            return (d.action, d.repeat, "")

        return ("dpad_down", 1,
                f"VETO press_a: focus is on '{d.focused_item}', which is "
                f"neither '{target}' nor a step toward it.")

    # -- the loop ----------------------------------------------------------
    def navigate_to(self, goal: str, target: str = "",
                    max_cycles: int | None = None,
                    label: str = "menu") -> dict[str, Any]:
        """Drive the menus until the goal is reached, aborted, or out of budget.

        `goal` is stated to the model in plain English ("start the level Sea of
        Sand in Chapter 1"). `target` is the item we expect to end up
        confirming, and is what the confirm gate checks focus against.

        Returns {"ok", "reason", "cycles", "last_screen", ...}. Never raises
        for a navigation failure - a launch that could not complete is a
        result to report, not an exception to crash on.
        """
        budget = int(max_cycles or self._cfg("max_cycles"))
        settle = float(self._cfg("settle_seconds"))
        timeout = float(self._cfg("llm_timeout_seconds"))

        try:
            camera = self.hardware.capture()
        except Exception as exc:
            return {"ok": False, "reason": f"Capture unavailable: {exc}",
                    "cycles": 0}

        print("\n" + "=" * 72, flush=True)
        print("  VISION-GUIDED MENU NAVIGATION", flush=True)
        print(f"  Goal   : {goal}", flush=True)
        print(f"  Target : {target or '(inferred from the goal)'}", flush=True)
        print(f"  Budget : {budget} cycles", flush=True)
        print("=" * 72, flush=True)

        history: list[str] = []
        last_screen = "unknown"
        # Per-navigation, so a retry cannot inherit the previous run's proof.
        self._confirmed_target = False
        # Per-cycle evidence for the launch section of the mechanics report.
        # The launch is the part a demo most wants to show, and it is
        # otherwise invisible once the console has moved on.
        self.steps: list[dict[str, Any]] = []

        for cycle in range(1, budget + 1):
            frame = camera.grab(allow_blank=True)
            if frame is None:
                return {"ok": False, "cycles": cycle,
                        "last_screen": last_screen,
                        "reason": ("Capture returned no frame - the device may "
                                   "have been taken by another application.")}

            saved = None
            if self.artifacts is not None:
                saved = self.artifacts.save_frame(frame, f"{label}-{cycle:02d}")

            b64 = self._encode(frame)
            if not b64:
                return {"ok": False, "cycles": cycle,
                        "last_screen": last_screen,
                        "reason": "Could not JPEG-encode the menu frame."}

            decision = self._read_screen(b64, goal, target, cycle, budget,
                                         history, timeout)
            if decision is None:
                continue
            d = decision
            last_screen = d.screen
            self.steps.append({
                "cycle": cycle,
                "screen": d.screen,
                "visible_items": list(d.visible_items),
                "focused_item": d.focused_item,
                "focus_evidence": d.focus_evidence,
                "reasoning": d.reasoning,
                "action": d.action,
                "repeat": d.repeat,
                "frame": saved,
            })

            outcome = self._handle(d, cycle, goal, target, settle, history,
                                   frame=frame)
            if outcome is not None:
                return outcome

        return {"ok": False, "cycles": budget, "last_screen": last_screen,
                "reason": (f"Did not reach the goal within {budget} menu "
                           f"cycles. Last screen seen: {last_screen}.")}

    def _read_screen(self, b64: str, goal: str, target: str, cycle: int,
                     budget: int, history: list[str],
                     timeout: float) -> MenuDecision | None:
        """One vision call. None means "retry next cycle", not "failed"."""
        from langchain_core.messages import HumanMessage

        recent = ("\n".join(f"  - {h}" for h in history[-5:])
                  or "  (nothing yet)")
        prompt = (
            f"{MENU_SYSTEM_PROMPT}\n"
            f"# This navigation\n"
            f"GOAL: {goal}\n"
            f"ITEM WE WANT TO CONFIRM: "
            f"{target or '(work it out from the goal)'}\n"
            f"Cycle {cycle} of {budget}\n\n"
            f"What has already been tried:\n{recent}\n\n"
            f"Read the attached screen and return the ONE next input."
        )
        messages = [HumanMessage(content=[
            {"type": "text", "text": prompt},
            {"type": "image_url",
             "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        ])]

        print(f"[menu {cycle:02d}] reading the screen ...", flush=True)
        try:
            d: MenuDecision = self._invoke_with_timeout(messages, timeout)
        except TimeoutError:
            print(f"  !! model did not answer within {timeout:.0f}s - "
                  f"retrying.", flush=True)
            history.append(f"cycle {cycle}: model timed out")
            return None
        except Exception as exc:
            print(f"  !! model error: {exc}", flush=True)
            history.append(f"cycle {cycle}: model error {exc}")
            time.sleep(1.0)
            return None

        print(f"  Screen  : {d.screen}", flush=True)
        if d.visible_items:
            print(f"  Items   : {d.visible_items}", flush=True)
        print(f"  Focused : {d.focused_item or '(none reported)'}"
              + (f"  [{d.focus_evidence}]" if d.focus_evidence else ""),
              flush=True)
        print(f"  Thinking: {d.reasoning}", flush=True)
        return d

    def _handle(self, d: MenuDecision, cycle: int, goal: str, target: str,
                settle: float, history: list[str],
                frame: np.ndarray | None = None) -> dict[str, Any] | None:
        """Act on one decision. A dict return ends navigation.

        `frame` is the screenshot the decision was made from, kept so a
        confirm can be verified against it rather than assumed.
        """
        if d.action == "done" or d.screen == "in_gameplay":
            # Gameplay starting is NOT proof the right level started. In the
            # live run a stray second A launched whatever the level list had
            # focused (Prologue) and the navigator still reported success -
            # so the caller could not tell a correct launch from a wrong one.
            # Say which level was actually confirmed, and admit when none was.
            confirmed = self._confirmed_target
            print("  >> gameplay reached."
                  + ("" if confirmed else
                     " NOTE: the target level was never explicitly confirmed, "
                     "so which level is running is UNVERIFIED."), flush=True)
            return {"ok": True, "cycles": cycle, "last_screen": d.screen,
                    "reason": ("Gameplay screen reached after confirming "
                               f"'{target or goal}'." if confirmed else
                               "Gameplay screen reached, but the target level "
                               "was never confirmed - the running level is "
                               "unverified."),
                    "target_confirmed": confirmed,
                    "focused_item": d.focused_item}

        if d.action == "abort" or d.screen == "store_or_purchase":
            reason = d.reasoning or f"Aborted on a {d.screen} screen."
            print(f"  >> ABORT: {reason}", flush=True)
            return {"ok": False, "cycles": cycle, "last_screen": d.screen,
                    "reason": reason, "aborted": True}

        if d.action == "wait":
            print(f"  Act     : wait {settle:.1f}s", flush=True)
            history.append(f"cycle {cycle}: {d.screen}, waited")
            time.sleep(settle)
            return None

        action, repeat, veto = self._gate_confirm(d, target or goal)
        # Record what will ACTUALLY run, not what the model proposed - the
        # gate can substitute a move for a confirm, and the launch report
        # must show the real input plus why it was changed.
        if self.steps:
            self.steps[-1]["dispatched"] = action
            self.steps[-1]["dispatched_repeat"] = repeat
            self.steps[-1]["veto"] = veto
        if veto:
            print(f"  {veto}", flush=True)
        print(f"  Act     : {action}" + (f" x{repeat}" if repeat > 1 else ""),
              flush=True)

        # Record that the TARGET itself was confirmed - not a waypoint, not a
        # splash screen - so the final result can be honest about which level
        # is actually running.
        if (action == "press_a" and not veto
                and d.screen in ("level_picker", "chapter_list", "main_menu")
                and self._matches(d.focused_item, (target or goal).strip())):
            self._confirmed_target = True

        self._dispatch(action, repeat)

        note = ""
        if action == "press_a":
            # A confirm moves to a NEW screen. Decide nothing until the
            # pixels prove that move happened, otherwise the next cycle reads
            # the old screen and confirms the same item twice - which is how
            # a second A once slipped through to the level list and launched
            # the wrong level.
            changed, delta = self._wait_for_change(
                frame, float(self._cfg("confirm_timeout")))
            if changed:
                print(f"  Screen changed after confirm (delta {delta:.1f}).",
                      flush=True)
                note = " -> screen changed"
            else:
                # Nothing moved. The confirm did not take, so say so instead
                # of letting the model infer "must need another press".
                print(f"  !! screen did NOT change after confirm "
                      f"(max delta {delta:.1f}). Treating the confirm as "
                      f"having had no effect.", flush=True)
                note = (" -> NO CHANGE, so that confirm did nothing; do not "
                        "simply repeat it")
        elif action in ("dpad_up", "dpad_down", "dpad_left", "dpad_right",
                        "bumper_left", "bumper_right"):
            # A HIGHLIGHT MOVE must be verified too. This was the other half
            # of the stale-frame bug and it is what made the launcher confirm
            # the WRONG ROW: at cycle 10 the d-pad moved CONTINUE -> SELECT
            # LEVEL, but the next screenshot still showed CONTINUE, so the
            # model pressed down AGAIN. Focus was then on OPTIONS while the
            # loop believed it was on SELECT LEVEL, and the confirm opened
            # Options. The reverse - reading a move that has not landed yet -
            # is what "went extra down" in the previous run.
            moved, delta = self._wait_for_move(
                frame, float(self._cfg("move_timeout")))
            if moved:
                print(f"  Highlight moved ({delta * 100:.2f}% of the frame "
                      f"changed).", flush=True)
                note = " -> highlight moved"
            else:
                # Either the press did nothing (list end / not a menu) or the
                # highlight repaint was too subtle to measure. Both are worth
                # telling the model, because "press it again" is the wrong
                # conclusion in the first case.
                print(f"  !! highlight did not visibly move "
                      f"(only {delta * 100:.2f}% changed) - it may already be "
                      f"at the end of the list.", flush=True)
                note = (" -> NO VISIBLE MOVE, so the highlight may already be "
                        "at the list end; try the opposite direction or "
                        "confirm what is focused")
        else:
            time.sleep(settle)

        history.append(
            f"cycle {cycle}: {d.screen}, focus='{d.focused_item}', "
            f"did {action}" + (f" x{repeat}" if repeat > 1 else "")
            + (" (vetoed confirm)" if veto else "") + note)
        return None
