import sys, time
sys.path.insert(0, r'c:\Users\sinha\Desktop\InstrumentsStore\xboxArudino\Xbox-Automation-Python')
sys.path.insert(0, r'c:\Users\sinha\Desktop\InstrumentsStore\xboxArudino\Xbox-Automation-Python\test-controller')

from capture.capture import ScreenCapture
from test_controller import ConsolePad
import cv2, pytesseract

cap = ScreenCapture()
pad = ConsolePad()

before = cap.grab()
cv2.imwrite('scratch/before_stick.png', before)

print('Moving stick down...')
pad.stick('left_stick', 'down', duration=0.3, strength=0.8)
time.sleep(1.0)

after = cap.grab()
cv2.imwrite('scratch/after_stick.png', after)

diff = cv2.absdiff(before, after)
print('Screen delta after stick down:', diff.mean())
print('Text after:\n', pytesseract.image_to_string(after).strip())
