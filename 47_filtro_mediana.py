import cv2
import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt

def filtro_mediana(A):
    S= np.median(A.flatten())    
    return S


img = cv2.imread('imagenes/cameraman.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#Ruido impulsional
mat_noise = np.random.random(gray.shape); #creates a uniform random variable from 0 to 1 

sp_noise_white= np.uint8(np.where(mat_noise>=0.7, 255,0))
sp_noise_black= np.uint8(np.where(mat_noise>=0.2,  1,0))

noise_img = cv2.multiply(gray,sp_noise_black)
noise_img = cv2.add(noise_img,sp_noise_white)


#aplicamos el filtro mediana
filtro= ndi.generic_filter(noise_img,filtro_mediana, [11,11])

plt.figure()
plt.imshow(noise_img, cmap="gray")
plt.show()

plt.figure()
plt.imshow(filtro, cmap="gray")
plt.show()

