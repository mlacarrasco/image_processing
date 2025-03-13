import cv2
import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt


def filtro_promedio_podado(subimage, acut):
    S= np.sort(subimage.flatten())
    B= S[acut:-acut]
    val=np.sum(B)/len(B)
    return val


img = cv2.imread('imagenes/lena.png')

gray = img[:,:,1]

#Ruido impulsional
mat_noise=np.random.random(gray.shape); #creates a uniform random variable from 0 to 1 

sp_noise_white= np.uint8(np.where(mat_noise>=0.8, 255,0))
sp_noise_black= np.uint8(np.where(mat_noise>=0.15,  1,0))
noise_img = cv2.multiply(gray,sp_noise_black)
noise_img = cv2.add(noise_img,sp_noise_white)

#aplicamos el filtro promedio alfa-acotado
filtro= ndi.generic_filter(noise_img,filtro_promedio_podado, [3,3], extra_keywords={'acut':3})

plt.figure()
plt.imshow(noise_img, cmap="gray")
plt.show()

plt.figure()
plt.imshow(filtro, cmap="gray")
plt.show()


