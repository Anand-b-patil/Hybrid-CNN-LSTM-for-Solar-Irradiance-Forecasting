import numpy as np
import cv2

# Create synthetic image (240x320 gradient)
img = np.linspace(0, 255, 240 * 320).reshape((240, 320)).astype(np.uint8)
img_color = cv2.applyColorMap(img, cv2.COLORMAP_JET)

# Save to test
cv2.imwrite("test_gradient.png", img_color)
