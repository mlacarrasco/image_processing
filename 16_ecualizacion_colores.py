import cv2
import numpy as np
import matplotlib.pyplot as plt

#lectura de la imagen
img = cv2.imread('imagenes/lena.png')
m,n,ch = img.shape

blue, green, red= cv2.split(img)

eq_blue  = cv2.equalizeHist(blue)
eq_green = cv2.equalizeHist(green)
eq_red   = cv2.equalizeHist(red)

#seteamos el output para tres canales de tipo uint8
out = cv2.merge([eq_red, eq_green, eq_blue])

#resultados
plt.imshow(img[:,:,[2,1,0]])
plt.title("Original")
plt.show()


plt.imshow(out)
plt.title("Ecualizada")
plt.show()