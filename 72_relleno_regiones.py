import cv2 
import numpy as np
from matplotlib.pyplot import figure, draw, pause

#imagen original

img = cv2.imread('imagenes/test.png')
A = img[:,:,0]
AC= cv2.bitwise_not(A).astype('uint8')
cv2.imshow('imagen complemento', AC)
cv2.waitKey(0)

#kernel
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))

#creacion imagen
m,n = AC.shape
M = np.zeros([m,n], dtype=np.uint8)
BIN= np.zeros([m,n], dtype=np.uint8)

#inicializacion
ind = np.where(AC==255)         #buscamos un valor donde img sea blanco
BIN[ind[0], ind[1]]     = 255
M[ind[0][0], ind[1][0]] =255     #inicializamos un valor en 255


fg = figure()
ax = fg.gca()
h = ax.imshow(M, cmap="gray")  # set initial display dimensions

for i in range(0,300):
    M = cv2.bitwise_and(cv2.dilate(M, kernel), AC)
    h.set_data(M)
    ax.set_title("indice= %i" %i)
    draw()
    pause(1e-3)

K= cv2.bitwise_not(M).astype('uint8')
R=cv2.bitwise_and(K, BIN)


cv2.imshow('Imagen Binaria',AC)
cv2.imshow('Relleno de regiones',R)

cv2.waitKey(0)

