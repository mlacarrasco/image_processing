import cv2
from skimage import filters
import matplotlib.pyplot as plt

#lectura de la imagen
img = cv2.imread('circuit.png')

#transformación a escala de grises
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#bordes
edges= filters.prewitt(gray)

#normalización
out = cv2.normalize(edges.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)

plt.imshow(out, cmap="gray")
plt.show()
