import cv2
import numpy as np
import matplotlib.pyplot as plt

#lectura de la imagen
img = cv2.imread('imagenes/cameraman.png', cv2.IMREAD_GRAYSCALE)
m,n = img.shape

#incorporamos ruido a la imagen original
mat_noise= np.random.rand(m,n)  #Agregamos ruido aleatorio entre 0 y 1
sp_noise_white= np.uint8(np.where(mat_noise>=0.99, 255,0))
sp_noise_black= np.uint8(np.where(mat_noise>=0.01,   1,0))
noise_img    = cv2.multiply(img,sp_noise_black)
noise_img_sp = cv2.add(noise_img,sp_noise_white)


#aplicamos el filtro
img_median   = cv2.medianBlur(noise_img_sp,3)

plt.imshow(img_median, cmap="gray")
plt.show()
