import cv2
import numpy as np

img = cv2.imread('lena.png')

# MORPH_CROSS, MORPH_ELLIPSE, MORPH_RECT
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(8,8))
print(kernel)
dilate = cv2.erode(img,kernel)

cv2.imshow('Erosion',dilate)
cv2.waitKey(0)

