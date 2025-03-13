import cv2
import matplotlib.pyplot as plt

#lectura imagen
img = cv2.imread('rice.png')

#filtros lineales
lpl = cv2.Laplacian(img,cv2.CV_8U)
uSharp= cv2.subtract(img, lpl)

plt.imshow(uSharp, cmap="gray")
plt.show()
