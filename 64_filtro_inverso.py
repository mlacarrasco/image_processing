import cv2
import numpy as np
import matplotlib.pyplot as plt

def filtro_inv(g, radio):
    G=np.fft.fft2(g)
    
    M,N = G.shape
    F= btw(M,N,2.1,radio)
 
    #limitamos el filtro H
    Hn=  F
    Ft=  G/Hn

    S= np.real(np.fft.ifft2(Ft))
    return S

def ruido_turbulento(f,k):

    F=np.fft.fft2(f)
    m,n =F.shape

    rx = np.linspace(-m/2, m/2, m)
    ry = np.linspace(-n/2, n/2, n)
    U,V = np.meshgrid(rx, ry)

    #ecuacion de turbulencia
    H= np.exp(-k*(U**2+V**2)**(5/6))
    H= np.fft.fftshift(H)

    #aplicacion del filtro
    FT=H*F 
    ft=np.real(np.fft.ifft2(FT))
    return ft


def btw(M,N,n,fc):
 
# M: filas de la imagen
# N: columnas de la imagen
# n: orden del filtro
# fc: frecuencia de corte
# Calculo de la malla
    vx = np.linspace(-M/2, M/2, M)
    vy = np.linspace(-N/2, N/2, N)
    U,V = np.meshgrid(vy,vx)
    f   = np.sqrt(U**2+V**2)
  
#filtro de butterworth centrado
    F =  1/(1+ (f/fc)**(2*n))

    F= np.fft.fftshift(F)
# filtro de butterwoth no centrado
      
    return(F)

#programa ppal
img  = cv2.imread('imagenes/cameraman.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray = cv2.normalize(gray.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)


im_ruido = ruido_turbulento(gray,0.005)
im_ruido = cv2.normalize(im_ruido, None, 0.0, 1.0, cv2.NORM_MINMAX)
plt.imshow(im_ruido, cmap='gray')
plt.show()

radio     = 26
output    = filtro_inv(im_ruido,radio)
plt.imshow(output, cmap='gray')
plt.show()
