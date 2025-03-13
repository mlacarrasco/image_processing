import cv2
import matplotlib.pyplot as plt

#lectura de la imagen
img = cv2.imread('cameraman.png')

#filtro no lineal (mediana)
img_median = cv2.medianBlur(img,7)


plt.imshow(img_median, cmap="gray")
plt.show()
