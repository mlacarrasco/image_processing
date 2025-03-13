import cv2
import matplotlib.pyplot as plt

#lectura de la imagen
img = cv2.imread('imagenes/lena.png') 

#acceso a posiciones de la imagen
roi = img[255:281,314:348,2]
subroi = roi[10:20, 15:25]

#binarizacion
rt,bina = cv2.threshold(subroi,150,255,cv2.THRESH_BINARY)

plt.imshow(bina, cmap="gray")
plt.show()
