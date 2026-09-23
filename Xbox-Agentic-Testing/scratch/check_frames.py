import cv2
import numpy as np

for name in ['step-000_0-before-capture_frame.png', '0-after-capture_frame.png', 'game-tile-1.png']:
    img = cv2.imread(f'artifacts/runs/run-20260904-222650/frames/{name}')
    prologue = img[580:630, 280:550]
    anotherland = img[640:695, 280:550]
    print(name)
    print('  Prologue mean:', np.mean(prologue))
    print('  Anotherland mean:', np.mean(anotherland))
