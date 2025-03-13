
import cv2
import numpy as np
import numpy.matlib
from matplotlib import cm
from matplotlib.pyplot import figure, draw, pause

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

cv2.imshow('Ruido Movimiento', noise_image)
cv2.waitKey(0)

fg = figure()
ax = fg.gca()

 #aplicamos el filtro de parametrico
h = ax.imshow(noise_image, cmap="gray")  # set initial display dimensions

m= 256
n= 256 
G = np.fft.fft2(noise_image)

p=np.array([[0 ,-1 ,0],[ -1, 4, -1],[0, -1, 0]])
Pp=np.fft.fft2(p,s=[m,n])

vector= np.linspace(1e-20,1e-5, 100)
for gamma in vector:
    
    F=(np.conj(H)*G)/(abs(H)**2+gamma*abs(Pp)**2)

    iRestored = np.real(np.fft.ifft2(F))
    h.set_data(iRestored)
    ax.set_title("value= %g" %gamma)
    draw()
    pause(0.1)

   


#cv2.imshow('Imagen Restaurada', iRestored)
#cv2.waitKey(0)
