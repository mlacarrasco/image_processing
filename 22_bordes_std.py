import cv2
import matplotlib.pyplot as plt

#lectua de la imagen
img = cv2.imread('cameraman.png', cv2.IMREAD_GRAYSCALE)

#normalización de la imagen
imgN = cv2.normalize(img.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)

#promedios
imgA = cv2.blur(imgN,(3,3))
imgB = cv2.blur(imgN*imgN,(3,3))

out = imgB - imgA**2

#normalizacion 0-1
out = cv2.normalize(out.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)

plt.imshow(out, cmap="gray")
plt.show()
