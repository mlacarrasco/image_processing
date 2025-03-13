
import cv2
import numpy as np
import numpy.matlib
from matplotlib import cm
from matplotlib.pyplot import figure, draw, pause, imshow, show


def ruido_movimiento(f,a,b,T):
 
  F =np.fft.fft2(f)   # Espectro
  m,n = F.shape
  
  rx  = np.linspace(-1, 1, m)
  ry  = np.linspace(-1, 1, n)
  U,V = np.meshgrid(rx, rx)

  #ecuacion de movimiento
  UV=U*a+V*b
        
  H= T * np.sinc(np.pi*UV)*np.exp(-1j*np.pi*UV)
  H= np.fft.fftshift(H)

  #aplicacion del filtro
  FT = H*F
  ft = np.real(np.fft.ifft2(FT))
  
  return ft, H


img = cv2.imread('imagenes/cameraman.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray = cv2.normalize(gray.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)

a=7
b=0
T=1
f= gray

noise_image, H = ruido_movimiento(f,a,b,T)
#cv2.imshow('Ruido Movimiento', noise_image)

fg = figure()
ax = fg.gca()
ax.imshow(noise_image, cmap='gray')
show()

fg = figure()
ax = fg.gca()

 #aplicamos el filtro de wienner
h = ax.imshow(noise_image, cmap='gray')  # set initial display dimensions

vector= np.linspace(1e-18,1e-3, 100)
for K in vector:

    W = np.conj(H)/(np.abs(H)**2 + K)
    G = np.fft.fft2(noise_image)
    F = W*G
    iRestored = np.real(np.fft.ifft2(F))
    h.set_data(iRestored)
    
    ax.set_title("value= %g" %K)
    draw()
    pause(0.05)

   
cv2.imshow('Imagen Restaurada', iRestored)
cv2.waitKey(0)
