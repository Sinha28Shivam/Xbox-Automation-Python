"""Which configured OCR engines load on this machine? No hardware needed.

    python dev\\probes\\check_ocr.py
    python dev\\probes\\check_ocr.py path\\to\\frame.png   # also read a frame
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from core.config import Config  # noqa: E402
from tools.vision_tools import _ocr, warm_ocr_impl  # noqa: E402


class _Ctx:
    settings = Config.load(ROOT / "config" / "settings.yaml", base=ROOT)
    scratch: dict = {}


ctx = _Ctx()
print("configured engines:", ctx.settings.list_of("verification.ocr.engines"))
result = warm_ocr_impl(ctx)  # type: ignore[arg-type]
print("ready:", result.get("engines"), "| errors:", result.get("errors") or result.get("error"))
if len(sys.argv) > 1:
    text, engine, error = _ocr(ctx, sys.argv[1])  # type: ignore[arg-type]
    print(f"\n--- {engine or 'no engine'} read {len((text or '').splitlines())} lines {error}")
    print((text or "")[:800])
sys.exit(0 if result.get("ok") else 1)
