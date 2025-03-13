import cv2
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.image as mpimg

def btw_pts(x,y,M,N,n,fc):
# Aplicacion del filtro Butterworth 
# en una posicion especifica

# x: posicion en x (fila)
# y: posicion en y (columna)
# M: filas de la imagen
# N: columnas de la imagen
# n: orden del filtro
# fc: frecuencia de corte
# Calculo de la malla
 
    vx=np.linspace(-M/2, M/2, M)
    vy=np.linspace(-N/2, N/2, N)
    U,V=np.meshgrid(vy,vx)
    f= np.sqrt(U**2+V**2)

    T=np.ones([M,N])
    # para cada punto de la lista
    print(len(x))
    for i in range(0, len(x)):
        
        #filtro de butterworth centrado
        H = 1- 1/(1+ (f/fc)**(2*n))
     
        # Movemos el punto del centro
        # a la posicion deseada
        a= x[i]- M/2
        b= y[i]- N/2
        H = np.roll(H, int(a), axis=0)
        H = np.roll(H, int(b), axis=1)

        T= T*H

    #Convertimos el filtro
    res= 1-T
    return (res)

#programa principal
img = cv2.imread('imagenes/noiseball.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray = cv2.normalize(gray.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)


m, n = gray.shape


# PASO 1)
# determinamos el espectro de la imagen
F = np.fft.fft2(gray)
FS = np.fft.fftshift(F)

# PASO 2)
# Parametros de Butterworth
od = 2 #orden del filtro
fc = 7 #frecuencia de corte (ancho)

# Funcion que genera puntos artificiales
# segun las coordenadas seleccionadas
x  = np.array([158,143,158,354,370,355])/2
y  = np.array([271,320,341,369,321,299])/2
HB = btw_pts(x,y,m,n,od,fc)

# Aplicamos el filtro sobre la imagen
ef= FS*HB

# Determinamos la imagen del ruido
nu = np.real(np.fft.ifft2(np.fft.fftshift(ef)))

# PASO 3)
# Determinamos el factor de compensación del ruido
a = 4
b = 4
w = np.zeros([m, n])

#agrandamos la imagen original
m_g  =  np.zeros([m+2*a, n+2*a])
m_nu =  np.zeros([m+2*a, n+2*a])

#incorporamos la informacion
m_g[a:-a, b:-b]=gray
m_nu[a:-a, b:-b]=nu
print(m_g.shape)

for i in range(a, m-a):
    for j in range(b, n-b):
        wg= np.zeros([2*a, 2*b])
        wn= np.zeros([2*a, 2*b])
       
        #ventanas
        wg  = m_g[-a+i:a+i,-b+j:b+j]
        wn  = m_nu[-a+i:a+i,-b+j:b+j]

        num = np.mean(wg*wn) - np.mean(wg)*np.mean(wn)
        den = np.mean(wn**2)- np.mean(wn)**2
        w[i-a, j-b]= num/den

# PASO 4)
# aplicamos la corrección del ruido con el factor de compensación
res= gray - w*nu

#borramos los bordes de la imagen de salida
output = res[2*a:-2*a, 2*b:-2*b]

fig = plt.figure(figsize=(16,6))
plt.subplot(121)
plt.imshow(gray, cmap="gray")
plt.subplot(122)
plt.imshow(output, cmap="gray")

plt.show()


