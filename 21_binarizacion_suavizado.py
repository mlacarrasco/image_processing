import cv2
import matplotlib.pyplot as plt

#lectura de la imagen
I = cv2.imread('rice.png', cv2.IMREAD_GRAYSCALE)

#filtro lineal: promedio
J = cv2.blur(I,(5,5))

#binarización
ret, BJ= cv2.threshold(J, 120, 255, type=cv2.THRESH_BINARY)

plt.imshow(BJ, cmap="gray")
plt.show()
