import cv2
import matplotlib.pyplot as plt

#lectura de la imagen
lena= cv2.imread('lena.jpg')
barbara= cv2.imread('barbara_gray.bmp')

#transformación a niveles de gris
gray1 = cv2.cvtColor(lena, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(barbara, cv2.COLOR_BGR2GRAY)

#suma
suma = cv2.add(gray1,gray2)

plt.imshow(suma, cmap="gray")
plt.show()
