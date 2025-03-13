import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

img = cv2.imread('imagenes/cameraman.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

F = np.fft.fft2(gray)
FS = np.fft.fftshift(F)

fa=10  # Frecuencia de corte
fc=28  # Frecuencia de corte
fb=30  # Frecuencia de corte


N= 3   # orden del filtro

x= np.linspace(-2*fc, 2*fc, 256)
y= np.linspace(-2*fc, 2*fc, 256)
X, Y = np.meshgrid(x,y)

H1 = 1/ (1+(np.power(np.sqrt(np.power(X,2)+np.power(Y,2))/fa,2*N)))
H2 = 1- 1/ (1+(np.power(np.sqrt(np.power(X,2)+np.power(Y,2))/fb,2*N)))

Z = H1 * H2
fig = plt.figure()
ax  = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap=cm.Spectral)
plt.show()


BF = Z * FS

S= np.fft.ifft2(np.fft.fftshift(BF))
out = cv2.normalize(abs(S), None, 0.0, 1.0, cv2.NORM_MINMAX)

plt.imshow(out,cmap="gray")
plt.show()