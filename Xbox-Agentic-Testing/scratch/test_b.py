import sys, time
from pathlib import Path
_HW = Path(__file__).resolve().parents[2] / "Xbox-Automation-Python"
sys.path.insert(0, str(_HW))
sys.path.insert(0, str(_HW / "test-controller"))

from capture.capture import ScreenCapture
from test_controller import ConsolePad
import cv2, pytesseract

cap = ScreenCapture()
pad = ConsolePad()

before = cap.grab()
pad.press('b')
time.sleep(1.0)
after = cap.grab()
diff = cv2.absdiff(before, after)
print('Screen delta after B:', diff.mean())
print('Text after:\n', pytesseract.image_to_string(after).strip())
