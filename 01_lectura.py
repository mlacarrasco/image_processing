import cv2
import matplotlib.pyplot as plt

#lectura de imagen
img = cv2.imread('imagenes/lena.png',0)

plt.imshow(img)
plt.show()
