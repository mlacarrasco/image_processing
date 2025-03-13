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
        bw[px:px+s, py:py+s]=255     

    return bw

bw = puntos_aleatorios(400, 80,10).astype('uint8')
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(25,25))

cv2.imshow('Original',bw)
cv2.waitKey(0)

tmp = cv2.dilate(bw,kernel)
cv2.imshow('Dilatada',tmp)
cv2.waitKey(0)


close = cv2.erode(tmp,kernel)

cv2.imshow('Clausura',close)
cv2.waitKey(0)

