import cv2, pytesseract
img = cv2.imread('scratch/live_screen.png')
crop = img[500:950, 250:700]
print(pytesseract.image_to_string(crop))
