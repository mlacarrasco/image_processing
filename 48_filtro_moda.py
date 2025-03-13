import cv2
import numpy as np
import scipy.ndimage as ndi
import statistics as sts
import matplotlib.pyplot as plt

def filtro_moda(A):
    try:
        S= A.flatten()
        md=sts.mode(S)
    except:
        md=0

    return md

img = cv2.imread('imagenes/cameraman.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#Ruido impulsional
mat_noise = np.random.random(gray.shape); #creates a uniform random variable from 0 to 1 

sp_noise_white= np.uint8(np.where(mat_noise>=0.95, 255,0))
sp_noise_black= np.uint8(np.where(mat_noise>=0.05,  1,0))
noise_img = cv2.multiply(gray,sp_noise_black)
noise_img = cv2.add(noise_img,sp_noise_white)

#normalizamos la imagen a rango de 0 a 255
#noise_img = cv2.normalize(noise_img.astype('float'), None, 0, 255, cv2.NORM_MINMAX)

#aplicamos el filtro moda
filtro= ndi.generic_filter(noise_img,filtro_moda, [3,3])

plt.figure()
plt.imshow(noise_img, cmap="gray")
plt.show()

plt.figure()
plt.imshow(filtro, cmap="gray")
plt.show()


