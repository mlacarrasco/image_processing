import cv2
import numpy as np

img = cv2.imread('../fotos/j.png')

# MORPH_CROSS, MORPH_ELLIPSE, MORPH_RECT
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(15,15))
print(kernel)
resultado = cv2.erode(img,kernel)

cv2.imshow('Original',img)
cv2.waitKey(0)

cv2.imshow('Erosion',resultado)
cv2.waitKey(0)

