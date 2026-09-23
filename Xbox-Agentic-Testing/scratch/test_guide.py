import sys, time
sys.path.insert(0, r'c:\Users\sinha\Desktop\InstrumentsStore\xboxArudino\Xbox-Automation-Python')
sys.path.insert(0, r'c:\Users\sinha\Desktop\InstrumentsStore\xboxArudino\Xbox-Automation-Python\test-controller')

from capture.capture import ScreenCapture
from test_controller import ConsolePad
import cv2, pytesseract

cap = ScreenCapture()
pad = ConsolePad()

before = cap.grab()
print('Pressing guide...')
pad.press('guide', duration=0.15)
time.sleep(1.5)
after = cap.grab()
cv2.imwrite('scratch/after_guide.png', after)
diff = cv2.absdiff(before, after)
print('Delta after guide:', diff.mean())
print('OCR after guide:\n', repr(pytesseract.image_to_string(after).strip()))
