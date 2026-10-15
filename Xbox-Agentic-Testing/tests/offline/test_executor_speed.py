"""Executor speed paths must not lose evidence.

1. Observe-only steps (capture/read/check) skip the before-frame, stability
   wait and diff, but still carry their frame and OCR text as evidence.
2. Input steps keep the full before/after/diff/OCR wrapper.
3. OCR results are cached per frame file and invalidated when it changes.
"""

import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agents.executor_agent import ExecutorAgent
from core.schemas import EvidenceKind, PlannedStep

calls: list[str] = []


class _Scratch:
    scratch = {"last_frame_path": "frame-7.png"}


class _Tools:
    context = _Scratch()


class _Ctx:
    tools = _Tools()

    class settings:
        @staticmethod
        def get(key, default=None):
            return default


agent = ExecutorAgent.__new__(ExecutorAgent)
agent.context = _Ctx()
agent.role = "executor"


def _call_tool(name, **kw):
    calls.append(name)
    if name == "capture_frame":
        return {"ok": True, "frame_path": "frame-7.png"}
    if name == "read_screen_text":
        return {"ok": True, "text": "Game activity\nStats", "engine": "pytesseract",
                "frame_path": kw.get("frame_path")}
    if name == "wait_for_stable_screen":
        return {"ok": True, "frame_path": "after.png"}
    if name == "verify_screen_changed":
        return {"ok": True, "delta": 12.0, "changed": True}
    if name == "press_button":
        return {"ok": True, "dispatched": True}
    return {"ok": True}


agent.call_tool = _call_tool
agent._dispatch = lambda step, dry: _call_tool(step.action, **step.arguments)


def _step(action, **args):
    return PlannedStep(index=0, action=action, arguments=args,
                       intent="x", expected_observation="x")


# 1. read_screen_text: one tool call, its own text is the evidence.
calls.clear()
r = agent._execute_step(_step("read_screen_text"), dry_run=False)
assert calls == ["read_screen_text"], calls
assert r.ocr_text == "Game activity\nStats" and r.frame_after == "frame-7.png"
assert any(e.kind == EvidenceKind.OCR_TEXT for e in r.evidence)

# 2. capture_frame: no before-frame / wait / diff; OCR still attached.
calls.clear()
r = agent._execute_step(_step("capture_frame"), dry_run=False)
assert "wait_for_stable_screen" not in calls and "verify_screen_changed" not in calls, calls
assert calls.count("capture_frame") == 1, calls
assert r.ocr_text and r.frame_after == "frame-7.png"

# 3. press_button keeps the full wrapper (before, stable, diff, OCR).
calls.clear()
r = agent._execute_step(_step("press_button", button="right"), dry_run=False)
for needed in ("capture_frame", "press_button", "wait_for_stable_screen",
               "verify_screen_changed", "read_screen_text"):
    assert needed in calls, (needed, calls)
assert r.screen_delta == 12.0

# 4. OCR cache: hit on same file, miss after the file changes.
from core.config import Config
import tools.vision_tools as vt

runs = {"n": 0}
vt._ocr_uncached = lambda ctx, path: (runs.__setitem__("n", runs["n"] + 1)
                                      or (f"text{runs['n']}", "fake", ""))


class _OcrCtx:
    settings = Config.load(Path(__file__).resolve().parents[2] / "config" / "settings.yaml")

    def __init__(self):
        self.scratch = {}


ctx = _OcrCtx()
frame = Path(tempfile.mkdtemp()) / "f.png"
frame.write_bytes(b"a")
assert vt._ocr(ctx, str(frame))[0] == "text1"
assert vt._ocr(ctx, str(frame))[0] == "text1" and runs["n"] == 1, "second read must hit cache"
time.sleep(0.02)
frame.write_bytes(b"bb")
assert vt._ocr(ctx, str(frame))[0] == "text2", "changed file must be re-read"

print("executor speed tests passed (4 cases)")
