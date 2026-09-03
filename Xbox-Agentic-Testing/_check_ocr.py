"""
_check_ocr.py - is OCR actually usable on this machine?

    python _check_ocr.py

WHY A SEPARATE CHECK
--------------------
`import pytesseract` succeeding proves nothing. It is a thin wrapper around the
tesseract EXECUTABLE, which is a separate install. The import works, the first
real call fails, and the failure surfaces in the middle of a hardware run
rather than up front.

PaddleOCR is the opposite trade: no external binary, but a large download on
first use and a slow model load. Both are optional here - the framework falls
back to frame differencing - but "optional" should be a decision you make, not
something you discover mid-run.

This script also OCRs a real captured frame, because the only question that
matters is whether OCR can read the Xbox UI on this rig. Game interfaces use
stylised fonts over animated backgrounds; an engine that reads documents
perfectly can still return nothing useful here.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LINE = "=" * 72


def main() -> int:
    print(f"\n{LINE}\n  OCR AVAILABILITY\n{LINE}\n")

    tesseract_ok = _check_tesseract()
    paddle_ok = _check_paddle()

    print(f"\n{LINE}\n  READING A REAL CAPTURED FRAME\n{LINE}\n")
    frame = _newest_frame()
    if frame is None:
        print("  No captured frames found. Run a test first:")
        print("    python console.py run \"press A and confirm the screen changes\"")
    else:
        print(f"  Frame: {frame.name}\n")
        if tesseract_ok:
            _read_with_tesseract(frame)
        if paddle_ok:
            _read_with_paddle(frame)

    print(f"\n{LINE}\n  SUMMARY\n{LINE}\n")
    if tesseract_ok or paddle_ok:
        engines = [n for n, ok in
                   (("pytesseract", tesseract_ok), ("paddleocr", paddle_ok)) if ok]
        print(f"  OCR is available: {', '.join(engines)}")
        print("  Text-based success criteria can now be verified directly.")
    else:
        print("  No OCR engine is usable. The framework still works - it falls")
        print("  back to frame differencing and vision-model judgement - but")
        print("  `text_present` criteria cannot be checked, and the report will")
        print("  say so rather than quietly passing them.\n")
        print("  To enable OCR, pick ONE:\n")
        print("    A. Tesseract (fast, small, needs a binary)")
        print("       winget install --id UB-Mannheim.TesseractOCR")
        print("       pip install pytesseract pillow\n")
        print("    B. PaddleOCR (no binary, better on stylised fonts,")
        print("       ~200MB model downloaded on first use)")
        print("       pip install paddleocr paddlepaddle")
    print()
    return 0


# ===========================================================================
def _check_tesseract() -> bool:
    try:
        import pytesseract
    except ImportError:
        print("  [ -- ] pytesseract   not installed")
        print("         pip install pytesseract pillow")
        return False

    print("  [ OK ] pytesseract   python package installed")

    # The package is only a wrapper; the binary is what does the work.
    try:
        version = pytesseract.get_tesseract_version()
        print(f"  [ OK ] tesseract    binary v{version}")
        return True
    except Exception as exc:
        print(f"  [FAIL] tesseract    BINARY MISSING - the python package "
              f"cannot work alone")
        print(f"         {str(exc).splitlines()[0][:90]}")
        print(f"         winget install --id UB-Mannheim.TesseractOCR")
        return False


def _check_paddle() -> bool:
    try:
        import paddleocr  # noqa: F401
        print("  [ OK ] paddleocr    installed")
        return True
    except ImportError:
        print("  [ -- ] paddleocr    not installed")
        print("         pip install paddleocr paddlepaddle")
        return False
    except Exception as exc:
        print(f"  [FAIL] paddleocr    installed but broken: "
              f"{str(exc).splitlines()[0][:70]}")
        return False


def _newest_frame() -> Path | None:
    runs = ROOT / "artifacts" / "runs"
    diagnostics = ROOT / "artifacts" / "diagnostics"

    frames: list[Path] = []
    if runs.is_dir():
        frames += list(runs.glob("*/frames/*.png"))
    if diagnostics.is_dir():
        frames += list(diagnostics.glob("*.png"))
    if not frames:
        return None
    return max(frames, key=lambda p: p.stat().st_mtime)


def _read_with_tesseract(frame: Path) -> None:
    try:
        import pytesseract
        from PIL import Image
        text = pytesseract.image_to_string(Image.open(frame))
        _show("pytesseract", text)
    except Exception as exc:
        print(f"  pytesseract failed: {str(exc).splitlines()[0][:90]}")


def _read_with_paddle(frame: Path) -> None:
    try:
        from paddleocr import PaddleOCR
        print("  paddleocr: loading model (slow on first run) ...")
        # PaddleOCR's constructor keywords have churned across versions -
        # `show_log` was removed, `use_angle_cls` renamed. Mirror the same
        # fallback chain vision_tools.py uses so this diagnostic reflects
        # what the real tool actually does, not a stale call signature.
        engine = None
        for kwargs in ({"lang": "en", "use_textline_orientation": True,
                        "enable_mkldnn": False},
                       {"lang": "en", "use_angle_cls": True,
                        "enable_mkldnn": False},
                       {"lang": "en", "enable_mkldnn": False},
                       {"enable_mkldnn": False},
                       {"lang": "en", "use_textline_orientation": True},
                       {"lang": "en", "use_angle_cls": True},
                       {"lang": "en"},
                       {}):
            try:
                engine = PaddleOCR(**kwargs)
                break
            except (TypeError, ValueError):
                continue
        if engine is None:
            raise RuntimeError("PaddleOCR could not be constructed")

        raw = engine.predict(str(frame)) if hasattr(engine, "predict") \
            else engine.ocr(str(frame))
        lines: list[str] = []
        for page in (raw or []):
            if page is None:
                continue
            if isinstance(page, dict):
                lines.extend(str(t) for t in (page.get("rec_texts") or []))
                continue
            for entry in page:
                try:
                    lines.append(entry[1][0])
                except Exception:
                    continue
        _show("paddleocr", "\n".join(lines))
    except Exception as exc:
        print(f"  paddleocr failed: {str(exc).splitlines()[0][:90]}")


def _show(engine: str, text: str) -> None:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    print(f"  {engine}: {len(lines)} line(s) read")
    for line in lines[:12]:
        print(f"      {line[:70]}")
    if len(lines) > 12:
        print(f"      ... and {len(lines) - 12} more")
    if not lines:
        # Worth stating plainly: this is the documented weakness of OCR on
        # console UIs, not necessarily a broken install.
        print("      (nothing read - console UIs use stylised fonts over")
        print("       animated backgrounds, which OCR handles poorly)")


if __name__ == "__main__":
    sys.exit(main())
