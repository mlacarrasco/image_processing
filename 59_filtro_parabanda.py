import cv2
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from matplotlib import cm


img = cv2.imread('imagenes/cameraman.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray = cv2.normalize(gray.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)

V = np.linspace(0,1,num=256)
Y= 0.1*np.sin(2*np.pi*40*V)

M= np.matlib.repmat(Y,256,1)
M= M.transpose()

noise_img= np.add(M, gray)
noise_img= np.add(M.transpose(), noise_img)
plt.imshow(noise_img)
plt.plot()

F = np.fft.fft2(noise_img)
FS = np.fft.fftshift(F)

fa=30  # Frecuencia de corte
fc=256  # Frecuencia de corte
fb=50  # Frecuencia de corte

N= 3   # orden del filtro

x= np.linspace(-fc/2, fc/2, fc)
y= np.linspace(-fc/2, fc/2, fc)
X, Y = np.meshgrid(x,y)

H1 = 1/ (1+(np.power(np.sqrt( X**2+Y**2 )/fa,2*N)))
H2 = 1- 1/ (1+(np.power(np.sqrt( X**2+Y**2)/fb,2*N)))

H = 1- H1 * H2
H= H - np.min(H)
H= H / np.max(H)


BF = H*FS

spectrum = 0.1*np.log(1+np.abs(BF))
cv2.imshow('fft2', spectrum)
cv2.waitKey(0)

S= np.fft.ifft2(np.fft.fftshift(BF))
out = cv2.normalize(abs(S), None, 0.0, 1.0, cv2.NORM_MINMAX)

plt.imshow(out)
plt.plot()