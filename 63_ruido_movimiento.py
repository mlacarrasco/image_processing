
import cv2
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from matplotlib import cm

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
  
  return ft


img = cv2.imread('imagenes/cameraman.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray = cv2.normalize(gray.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)

a=2
b=1

T=1
f= gray

output = ruido_movimiento(f,a,b,T)

plt.imshow(output, cmap='gray')
plt.show()

