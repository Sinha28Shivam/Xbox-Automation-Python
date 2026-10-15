"""Offline regression tests for guided_navigate_to_text (no Xbox required).

Each case reproduces a failure seen on real hardware in run-20261014-162340.
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.artifacts import ArtifactStore
from tools.registry import ToolContext, make_tool, ok
import tools.navigation_tools as nav

# A fake console: `screens[i]` is what is shown after i presses, and
# `focus[i]` is the item the green highlight sits on.
console = {"presses": 0, "screens": [], "focus": [], "settle_calls": 0}


class _Controls:
    def resolve(self, name):
        return ("button", name)


class _Pad:
    def press_times(self, name, *args):
        console["presses"] += 1
        console["last"] = name
        return True


class _Hardware:
    controls = _Controls()

    def pad(self, console_name=None):
        return _Pad()


class _Settings:
    def get(self, key, default=None):
        return default


def _frame(label="f", **_):
    return ok(frame_path=f"frame-{console['presses']}.png")


def _stable(**_):
    console["settle_calls"] += 1
    return ok(frame_path=f"frame-{console['presses']}.png")


def _screen_index(frame_path):
    return min(int(frame_path.split("-")[1].split(".")[0]),
               len(console["screens"]) - 1)


nav._capture_frame = lambda ctx: make_tool(_frame, "capture_frame", "fake")
nav._wait_for_stable_screen = lambda ctx: make_tool(_stable, "wait", "fake")
nav.read_screen_text_impl = lambda ctx, frame_path=None: ok(
    text=console["screens"][_screen_index(frame_path)])


def _fake_focus(ctx, frame_path=None, expected_label=""):
    focused = console["focus"][_screen_index(frame_path)]
    hit = expected_label.lower() in focused.lower()
    return ok(matched=hit, selected_label=focused,
              highlight_bbox={"x": 0, "y": 0, "width": 1, "height": 1}) if hit \
        else {"ok": False, "observed_text": focused,
              "highlight_bbox": {"x": 0, "y": 0, "width": 1, "height": 1}}


nav.detect_focus_highlight_impl = _fake_focus


def _run(screens, focus, **kwargs):
    console.update(presses=0, screens=screens, focus=focus, settle_calls=0)
    ctx = ToolContext(hardware=_Hardware(), settings=_Settings(),
                      artifacts=ArtifactStore(tempfile.mkdtemp(), "t"))
    return nav._guided_navigate_to_text(ctx).func(interval=0.0, **kwargs)


GAME_ACTIVITY_LIST = ("Game activity\nAchievements\nMore achievements\nStats\n"
                      "Friends who play\nGo to official club")

# 1. Tab heading reached after exactly 3 presses; heading is never focused.
result = _run(
    ["Home", "People", "Parties & chats", GAME_ACTIVITY_LIST],
    ["Home", "People", "Start a party", "Achievements"],
    target_text="Game Activity", direction="right", max_presses=3,
    require_focus=False)
assert result["ok"] and console["presses"] == 3, result
assert console["settle_calls"] >= 3, "must wait for a stable screen after each press"

# 2. "Stats" is VISIBLE on the first screen but focus is on Achievements.
#    The old tool stopped here and A opened Achievements. It must keep moving.
result = _run(
    [GAME_ACTIVITY_LIST] * 4,
    ["Achievements", "More achievements", "Stats", "Friends who play"],
    target_text="Stats", direction="down", max_presses=5)
assert result["ok"] and console["presses"] == 2, (console["presses"], result)
assert result["attempts"][0]["text_visible"] and not result["attempts"][0]["matched"]

# 3. Visible but never focused -> explicit failure, not a false success.
result = _run(
    [GAME_ACTIVITY_LIST] * 3, ["Achievements"] * 3,
    target_text="Stats", direction="down", max_presses=2)
assert not result["ok"] and result["text_seen"], result
assert "never became the highlighted item" in result["error"]

# 4. Target never appears -> failure that says so.
result = _run(["Home", "People", "Chats"], ["Home", "People", "Chats"],
              target_text="Game Activity", direction="right", max_presses=2,
              require_focus=False)
assert not result["ok"] and not result["text_seen"] and console["presses"] == 2

print("guided navigation tests passed (4 cases)")
