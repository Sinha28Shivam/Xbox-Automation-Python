import cv2
import pytesseract

img = cv2.imread('scratch/live_screen.png')
print('Frame shape:', img.shape)
data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
for i in range(len(data['text'])):
    w = data['text'][i].strip()
    if w:
        print(f"{w:20} conf={data['conf'][i]:3} box=({data['left'][i]}, {data['top'][i]}, {data['width'][i]}, {data['height'][i]})")
