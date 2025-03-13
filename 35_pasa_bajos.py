import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

img = cv2.imread('imagenes/cameraman.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

r=40
x= np.linspace(-127, 128,256)
y= np.linspace(-127, 128,256)
X, Y = np.meshgrid(x,y)

Z = np.sqrt(X**2+Y**2)<r

F = np.fft.fft2(gray)    #aplicar fourier a imagen original
FS = np.fft.fftshift(F)  #reordena los cuadrantes del espectro
BF = FS * Z              #multiplica el filtro Z por el espectro


#vemos el filtro en 3D
fig = plt.figure()
ax  = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap=cm.Spectral)
plt.show()


spectrum_filter = np.log(np.abs(FS)) * Z
out = cv2.normalize(spectrum_filter, None, 0.0, 1.0, cv2.NORM_MINMAX)
plt.imshow(out, cmap='gray')
plt.show()

EF = np.sum(abs(np.power(FS,2)))
EI = np.sum(abs(np.power(BF,2)))

ener = (1- ((EF-EI)/ EI))*100
print(ener)

S= np.fft.ifft2(np.fft.fftshift(BF))
out = cv2.normalize(abs(S), None, 0.0, 1.0, cv2.NORM_MINMAX)


plt.imshow(out,cmap="gray")
plt.show()
