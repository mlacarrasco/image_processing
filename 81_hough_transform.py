import cv2
import numpy as np
from skimage.transform import hough_line, hough_line_peaks
import matplotlib.pyplot as plt
from matplotlib import cm

#programa ppal
I = cv2.imread('imagenes/cuadrados.png', cv2.IMREAD_GRAYSCALE)
BW = cv2.Canny(I,10,150,apertureSize = 3)

angulos = np.linspace(-np.pi/2, np.pi/2, 360)
h, theta, d = hough_line(BW, theta=angulos)

# Mapa de acumulación
fig = plt.figure()
plt.imshow(np.log(1 + h),
             extent=[np.rad2deg(theta[-1]), np.rad2deg(theta[0]), d[-1], d[0]],
             cmap=cm.hot,aspect=1/25)
plt.xlabel('Angulo')
plt.ylabel('Acumulador')
plt.show()

fig = plt.figure()
plt.imshow(I, cmap='gray')
eje_x = np.array((0, BW.shape[1]))

valores_maximos= hough_line_peaks(h, theta, d, threshold=60)




for accum, theta, rho in zip(*valores_maximos):  #el * separa un arreglo
    y0 = (rho - eje_x[0] * np.cos(theta)) / np.sin(theta)
    y1 = (rho - eje_x[1] * np.cos(theta)) / np.sin(theta)
    plt.plot(eje_x, (y0, y1), '-r')


plt.xlim(eje_x)
plt.ylim((BW.shape[0], 0))
plt.title('Lineas de mapa de Hough')

plt.show()