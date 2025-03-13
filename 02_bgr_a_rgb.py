import cv2
import matplotlib.pyplot as plt

#lectura de imagen
img = cv2.imread('imagenes/lena.png')

#transformación a niveles de gris
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img)
plt.show()
