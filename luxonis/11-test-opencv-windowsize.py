import cv2
import numpy as np

while True:
    size_x = np.random.randint(640, 1280)
    size_y = np.random.randint(480, 720)
    frameRgb = np.random.randint(0, 255, (size_y, size_x, 3), dtype=np.uint8)
    cv2.imshow("rgb", frameRgb)

    if cv2.waitKey(1) == ord('q'):
        break