import cv2, pytesseract
img = cv2.imread(str(__import__('pathlib').Path(__file__).resolve().parents[2] / 'scratch' / 'live_screen.png'))
crop = img[500:950, 250:700]
print(pytesseract.image_to_string(crop))
