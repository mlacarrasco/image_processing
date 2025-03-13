
import cv2 
import numpy as np
from matplotlib.pyplot import figure, draw, pause

#imagen original

img = cv2.imread('imagenes/test.png')
A = img[:,:,0].astype('uint8')

#kernel
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))

#creacion imagen
m,n = A.shape
M = np.zeros([m,n], dtype=np.uint8)
BIN= np.zeros([m,n], dtype=np.uint8)

#inicializacion
ind= np.where(A==255)         #buscamos un valor donde img sea blanco
BIN[ind[0], ind[1]]   = 255
M[ind[0][0], ind[1][0]]=255     #inicializamos un valor en 255

fg = figure()
ax = fg.gca()
h = ax.imshow(M, cmap="gray")  # set initial display dimensions

for i in range(0,300):
    M= cv2.bitwise_and(cv2.dilate(M, kernel), BIN)
    h.set_data(M)
    ax.set_title("indice= %i" %i)
    draw()
    pause(1e-3)

cv2.imshow('Imagen Binaria',A)
cv2.imshow('Componentes conectadas',M)

cv2.waitKey(0)

