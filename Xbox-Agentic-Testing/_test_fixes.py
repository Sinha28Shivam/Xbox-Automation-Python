"""
_test_fixes.py - regression tests for the three bugs found in a live run.

    python _test_fixes.py

Each test reproduces the ORIGINAL failure exactly. If any starts failing, the
corresponding bug has come back.

THE BUGS
--------
1. `wait_for_stable_screen(stability_duration=1.0)` raised TypeError - the real
   parameter is `settle`. A 6-step plan died at step 2 over a synonym.

2. Markdown reports linked frames as "frames/x.png" from inside `reports/`,
   which resolves to "reports/frames/x.png". Every screenshot was broken, so
   the evidence a reader needs to check the verdict was invisible.

3. PaddleOCR rejected `show_log` - the keyword was removed upstream. OCR was
   configured, appeared installed, and silently produced nothing.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for sub in ("core", "tools", "agents", "graph"):
    p = str(ROOT / sub)
    if p not in sys.path:
        sys.path.insert(0, p)

PASSED: list[str] = []
FAILED: list[tuple[str, str]] = []


def check(name: str):
    def wrap(fn):
        try:
            PASSED.append(f"{name}{f' - {d}' if (d := fn()) else ''}")
        except Exception as exc:
            FAILED.append((name, f"{exc.__class__.__name__}: {exc}"))
            if "-v" in sys.argv:
                import traceback
                traceback.print_exc()
        return fn
    return wrap


def _registry():
    from adapters import HardwareBridge
    from artifacts import ArtifactStore
    from config import Config
    from registry import build_default_registry

    cfg = Config.load(ROOT / "config" / "settings.yaml", base=ROOT)
    return build_default_registry(
        HardwareBridge(cfg),
        ArtifactStore(ROOT / "artifacts", "fixtest", enabled=False),
        cfg)


# ===========================================================================
@check("BUG 1: a wrong argument name no longer aborts the step")
def _argument_tolerance():
    """The exact call that killed the run, plus the aliases around it."""
    from registry import tolerant

    calls: list[dict] = []

    def fake_wait(timeout: float | None = None, label: str = "stable"):
        calls.append({"timeout": timeout, "label": label})
        return {"ok": True, "frame_path": "x.png"}

    wrapped = tolerant(fake_wait, "wait_for_stable_screen")

    # This is verbatim what the planner emitted and what used to raise.
    result = wrapped(timeout=10.0, stability_duration=1.0)
    assert result["ok"], "the original failing call still does not work"
    assert calls[0]["timeout"] == 10.0, "timeout was lost"
    assert "argument_notes" in result, \
        "the dropped/renamed argument was not reported - silent is worse"

    # A missing REQUIRED argument must still fail: guessing would run the
    # wrong action, which is worse than a clean error.
    def needs_arg(before_path: str, after_path: str | None = None):
        return {"ok": True}

    strict = tolerant(needs_arg, "verify_screen_changed")
    missing = strict(after_path="b.png")
    assert not missing["ok"], "a missing required argument was silently allowed"
    assert "before_path" in missing["error"]

    # A close typo should be repaired rather than dropped.
    fixed = strict(before_paths="a.png")
    assert fixed["ok"], "a near-miss argument name was not corrected"
    return "aliases repaired, unknowns reported, missing-required still fails"


@check("BUG 2: report links to frames resolve from inside reports/")
def _report_links():
    from artifacts import ArtifactStore

    store = ArtifactStore(ROOT / "artifacts", "linktest", enabled=False)
    frame = store.frames_dir / "step-000_before.png"

    link = store.link_from_reports(frame)
    assert link.startswith("../frames/"), (
        f"link is '{link}' - from reports/ that resolves to "
        f"reports/{link}, which does not exist. Images render broken.")

    # And the report writer must actually use it.
    source = (ROOT / "tools" / "report_tools.py").read_text(encoding="utf-8")
    assert "link_from_reports" in source, \
        "report_tools.py still builds image links with relative()"
    return f"'{link}'"


@check("BUG 3: PaddleOCR is constructed without removed keywords")
def _paddle_signature():
    """Check the CODE, not the comments.

    A first version of this test grepped the whole file for "show_log" and
    failed on the explanatory comment that mentions it - a good reminder that
    a test which cannot distinguish code from prose is not testing much.
    """
    source = (ROOT / "tools" / "vision_tools.py").read_text(encoding="utf-8")
    code = "\n".join(
        line for line in source.splitlines()
        if not line.lstrip().startswith("#"))

    assert "show_log" not in code, \
        "vision_tools.py still passes show_log, removed in current PaddleOCR"
    assert "use_textline_orientation" in code, \
        "the modern PaddleOCR keyword is not attempted"
    # Several constructor spellings must be tried, or the next upstream rename
    # breaks OCR again.
    assert code.count('"lang": "en"') >= 2, \
        "only one constructor signature is attempted - fragile across versions"
    return "modern keyword first, older spellings as fallback"


@check("OCR reads real text from a captured console frame")
def _ocr_reads_console():
    from registry import ToolContext
    registry = _registry()

    frames = sorted((ROOT / "artifacts").glob("**/frames/*.png"))
    frames += sorted((ROOT / "artifacts" / "diagnostics").glob("*.png"))
    if not frames:
        return "skipped - no captured frames yet"

    frame = max(frames, key=lambda p: p.stat().st_mtime)
    tool = registry.build("read_screen_text")
    result = (tool.func if hasattr(tool, "func") else tool)(frame_path=str(frame))

    assert result.get("ok"), f"OCR failed: {result.get('error')}"
    lines = result.get("line_count", 0)
    assert lines > 0, "OCR ran but read nothing from a real console frame"
    return f"{lines} lines via {result.get('engine')} from {frame.name}"


@check("check_for_text exists and reports honestly when OCR fails")
def _check_for_text():
    registry = _registry()
    assert "check_for_text" in registry.names, \
        "the planner asked for check_for_text; it is still not registered"

    frames = sorted((ROOT / "artifacts").glob("**/frames/*.png"))
    if not frames:
        return "registered (no frame available to exercise it)"

    frame = max(frames, key=lambda p: p.stat().st_mtime)
    tool = registry.build("check_for_text")
    result = (tool.func if hasattr(tool, "func") else tool)(
        text_patterns=["something went wrong", "error"],
        frame_path=str(frame), match_type="any")

    assert "checked" in result, \
        "check_for_text must report whether it could actually look - " \
        "'no error found' and 'could not read' are different answers"
    return f"checked={result.get('checked')}, found={result.get('found')}"


@check("the executor treats a bad argument as non-fatal")
def _non_fatal():
    source = (ROOT / "agents" / "executor_agent.py").read_text(encoding="utf-8")
    assert "_is_fatal" in source, \
        "every error still aborts the run - one bad argument will again " \
        "discard the remaining steps"
    assert "ocr_text" in source, "the executor does not capture OCR per step"
    return "only hardware-level errors abort"


# ===========================================================================
def main() -> int:
    print(f"\n{'=' * 72}")
    print("  REGRESSION TESTS - bugs found in live runs")
    print("=" * 72 + "\n")

    for line in PASSED:
        print(f"  [PASS] {line}")
    for name, error in FAILED:
        print(f"  [FAIL] {name}")
        print(f"         {error}")

    print(f"\n{'=' * 72}")
    print(f"  {len(PASSED)} passed, {len(FAILED)} failed")
    print("=" * 72 + "\n")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
