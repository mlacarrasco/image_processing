import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('imagenes/cameraman.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

r=40
x= np.linspace(-127, 128,256)
y= np.linspace(-127, 128,256)
X, Y = np.meshgrid(x,y)

Z = np.sqrt(np.power(X,2)+np.power(Y,2))

F = np.fft.fft2(gray)
FS = np.fft.fftshift(F)
BF = FS * Z


S= np.fft.ifft2(np.fft.fftshift(BF))
out = cv2.normalize(abs(S), None, 0.0, 1.0, cv2.NORM_MINMAX)

plt.imshow(out,cmap="gray")
plt.show()