import cv2
import matplotlib.pyplot as plt
from skimage import filters

#lectura imagen
img = cv2.imread('cameraman.png')

#transformación a escala de grises
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#bordes
roberts = filters.roberts(gray)

plt.imshow(roberts, cmap="gray")
plt.show()
