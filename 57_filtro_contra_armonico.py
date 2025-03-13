import cv2
import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt 

def filtro_contra_armonico(A, Q):
    S= np.sum(A.flatten()**(Q+1)) / np.sum(A.flatten()**Q)
    return S
    

img = cv2.imread('imagenes/cameraman.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#Ruido impulsional
mat_noise=np.random.random(gray.shape); #creates a uniform random variable from 0 to 1 
sp_noise_black= np.uint8(np.where(mat_noise>=0.2,  1,0))
noise_img = cv2.multiply(gray,sp_noise_black)

#aplicamos el filtro promedio alfa-acotado
filtro= ndi.generic_filter(noise_img,filtro_contra_armonico, [3,3], extra_keywords={'Q':20})

plt.figure()
plt.imshow(noise_img, cmap="gray")
plt.show()

plt.figure()
plt.imshow(filtro, cmap="gray")
plt.show()


