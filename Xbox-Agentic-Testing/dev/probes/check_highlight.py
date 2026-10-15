import cv2
import numpy as np

img = cv2.imread('artifacts/runs/run-20260904-222650/frames/game-tile-1.png')

# Let's crop around Prologue and Anotherland
prologue_crop = img[580:630, 280:550]
anotherland_crop = img[640:695, 280:550]

print('Prologue mean color (BGR):', np.mean(prologue_crop, axis=(0,1)))
print('Anotherland mean color (BGR):', np.mean(anotherland_crop, axis=(0,1)))
print('Prologue max brightness:', np.max(prologue_crop))
print('Anotherland max brightness:', np.max(anotherland_crop))
