import cv2
import matplotlib.pyplot as plt

#lectura de imagen
img = cv2.imread('imagenes/cameraman.png')

#inverso
neg = 255-img

plt.imshow(neg)
plt.show()
