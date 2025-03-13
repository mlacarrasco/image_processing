import cv2
import numpy as np
import scipy.ndimage as ndi
from math import pi
import matplotlib.pyplot as plt 


def filtro_gaussiano(A):
    A= np.reshape(A, (10,10))
    t= 10
    sigma = 2.2
    ventana= np.linspace(-t/2, t/2, t)
    u,v = np.meshgrid(ventana, ventana)
    G= (1/(sigma**2*2*pi))*np.exp(-(u**2+v**2)/(2*sigma**2))
    N= G/np.sum(G.flatten())  #normalizamos
    T = N*A
    return np.sum(T)

    
img = cv2.imread('cameraman.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#Ruido impulsional
mat_noise=np.random.random(gray.shape); #creates a uniform random variable from 0 to 1 
sp_noise_black= np.uint8(np.where(mat_noise>=0.2,  1,0))
noise_img = cv2.multiply(gray,sp_noise_black)

#aplicamos el filtro promedio alfa-acotado
filtro= ndi.generic_filter(noise_img,filtro_gaussiano, [10,10])

plt.figure()
plt.imshow(noise_img, cmap="gray")
plt.show()

plt.figure()
plt.imshow(filtro, cmap="gray")
plt.show()

