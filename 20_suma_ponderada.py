import cv2
import matplotlib.pyplot as plt

#lectura de la imagen
lena= cv2.imread('/imagenes/lena.png')
barbara= cv2.imread('/imagenes/barbara_gray.bmp')

#transformación a niveles de gris
gray1 = cv2.cvtColor(lena, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(barbara, cv2.COLOR_BGR2GRAY)

#suma ponderada
suma_w = cv2.addWeighted(gray1, 0.3, gray2, 0.7, 0)

plt.imshow(suma_w, cmap="gray")
plt.show()
