import cv2
import numpy as np
from math import ceil

def puntos_aleatorios(L, N, delta):
    #% L: Taman de la imagen
    #% N: Numero de puntos
    #% delta: tamano maximo de los pixels 
    bw = np.zeros([L,L])

    #%tamaño del pixel
    spx=delta

    for i in range(0, N):
        px= ceil(np.random.rand(1)*(L-spx))
        py= ceil(np.random.rand(1)*(L-spx))
        s = ceil(np.random.rand(1)*spx)
        bw[px:px+s, py:py+s]=1      

    return bw

bw = puntos_aleatorios(400, 80,10)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))

#Etapa 1: Apertura
close = cv2.erode(bw,kernel)
AP = cv2.dilate(close,kernel)

#Etapa 2: Deteccion de Bordes
C = cv2.bitwise_not(cv2.erode(AP, kernel))
R = cv2.bitwise_and(AP, C)


cv2.imshow('Original',bw)
cv2.imshow('Clausura',R)
cv2.waitKey(0)

