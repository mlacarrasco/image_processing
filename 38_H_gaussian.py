
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

x= np.linspace(-127, 128,256)
y= np.linspace(-127, 128,256)
X, Y = np.meshgrid(x,y)

#creamos el filtro

H= np.exp(-0.0005*(X**2+Y**2))
UH = np.fft.fftshift(H)

#vemos el filtro en 3D
fig = plt.figure()
ax  = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, UH, cmap=cm.Spectral)
plt.show()


#vamos a leer la imagen
img   = cv2.imread('imagenes/cameraman.png')
gray  = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
F     = np.fft.fft2(gray)

FILT= UH*F

S    = np.fft.ifft2(np.fft.fftshift(FILT))
out = cv2.normalize(abs(S), None, 0.0, 1.0, cv2.NORM_MINMAX)


plt.imshow(out,cmap="gray")
plt.show()
