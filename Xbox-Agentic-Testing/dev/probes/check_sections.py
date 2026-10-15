import cv2, pytesseract
img = cv2.imread(str(__import__('pathlib').Path(__file__).resolve().parents[2] / 'scratch' / 'after_stick.png'))
# let's crop top, middle, bottom
top = img[:300, :]
mid = img[300:700, :]
bot = img[700:, :]
print('TOP:', pytesseract.image_to_string(top).strip())
print('MID:', pytesseract.image_to_string(mid).strip())
print('BOT:', pytesseract.image_to_string(bot).strip())
