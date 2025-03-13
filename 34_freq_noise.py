import cv2
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt

img = cv2.imread('cameraman.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray = cv2.normalize(gray.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)

#vamos a crear ruido
V = np.linspace(0,1,num=256)
Y = 0.2*np.sin(2*np.pi*10*V)
M = np.matlib.repmat(Y,256,1)
noise_img= np.add(M, gray)

#vamos aplicar fourier
F = np.fft.fft2(noise_img)
fshift = np.fft.fftshift(F)

#bloqueamos las frecuencias con ruido
fshift[122:130,118:120]=0.0
fshift[122:130,138:140]=0.0

S = np.fft.ifft2(np.fft.fftshift(fshift))

#desplegamos los resultados
fig, ax = plt.subplots(1,2)
fig.set_size_inches(10,10)
ax[0].imshow(abs(noise_img), cmap='gray')
ax[1].imshow(abs(S), cmap='gray')
plt.show()
