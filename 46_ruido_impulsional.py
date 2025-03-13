import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

img = cv2.imread('imagenes/cameraman.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#Ruido impulsional
mat_noise=np.random.random(gray.shape); #creates a uniform random variable from 0 to 1 

noise_img = gray.copy()  #copiamos la matriz original
sp_noise_white= mat_noise>=0.95 #matriz logica
sp_noise_black= mat_noise<=0.05 #matriz logica

noise_img[sp_noise_white] =255
noise_img[sp_noise_black] = 0


plt.figure()
plt.imshow(noise_img, cmap="gray")
plt.show()