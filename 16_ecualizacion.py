import cv2
import matplotlib.pyplot as plt

#lectura de imagen
img = cv2.imread('imagenes/lena.jpg')


#transformación a nivel de gris
src = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#ecualizacion
dst = cv2.equalizeHist(src)


plt.imshow(src, cmap="gray")
plt.title('Imagen original')
plt.show()

plt.imshow(dst, cmap="gray")
plt.title('Imagen ecualizada')
plt.show()
