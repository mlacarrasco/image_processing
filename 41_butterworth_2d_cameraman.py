import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

fc=28  # Frecuencia de corte
N= 3   # orden del filtro

x= np.linspace(-2*fc+1, 2*fc, 256)
y= np.linspace(-2*fc+1, 2*fc, 256)
X, Y = np.meshgrid(x,y)
print(X)


Z = 1/ (1+(np.power(np.sqrt(np.power(X,2)+np.power(Y,2))/fc,2*N)))
print (Z.shape)


fig = plt.figure()
ax  = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap=cm.Spectral)
plt.show()


img = cv2.imread('imagenes/cameraman.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

F = np.fft.fft2(gray)
FS = np.fft.fftshift(F)


fc=28  # Frecuencia de corte
N= 3   # orden del filtro

x= np.linspace(-2*fc, 2*fc, 256)
y= np.linspace(-2*fc, 2*fc, 256)
X, Y = np.meshgrid(x,y)

Z = 1/ (1+(np.power(np.sqrt(np.power(X,2)+np.power(Y,2))/fc,2*N)))
BF = Z * FS

S= np.fft.ifft2(np.fft.fftshift(BF))
out = cv2.normalize(abs(S), None, 0.0, 1.0, cv2.NORM_MINMAX)

plt.imshow(out,cmap="gray")
plt.show()