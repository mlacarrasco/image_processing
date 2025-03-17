"""
 Universidad Diego Portales
 Facultad de Ingeniería y Ciencias
 Procesamiento Digital de Imágenes
 Miguel Carrasco (miguel.carrasco@mail.udp.cl)

%rev.$1.2.$ date: 09/05/2020
%rev.$1.2.$ date: 26/05/2021
%rev.$1.3.$ date: 17/03/2025
"""

import  sys
import cv2
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib import cm
from haarmtx import haarmtx

def encode_wavelet(img, H, type):
    # basado en ideas de BlkProc (Matlab)
    # Author: Miguel Carrasco
    n= H.shape[0]
    rs, cs = img.shape
    enc = np.zeros_like(img)
    bimg= np.hstack((img, np.flip(img,1)))
    bimg =np.vstack((bimg, np.flip(bimg,0)))
    #haar encode
    for i in range(0,rs,n):
        for j in range(0,cs,n):
            tmp = bimg[i:i+n, j:j+n]
            if type==0:
                enc[i:i+n, j:j+n]= np.dot(np.dot(H,tmp), np.transpose(H))
            else:
                enc[i:i+n, j:j+n]= np.dot(np.dot(np.transpose(H),tmp), H )
    return(enc)


def  agrupar(A, f):
    n= A.shape[0]
    S = np.zeros([n,n])
    delta = n//2

    V,H= cuadrantes(delta)
    #haar decode
    for p in range(0,4):
        
        #caracteristicas    
        if (p==0 and f>2):
            #cuadrante PPAL
            tmp = A[V[p]:V[p]+delta, H[p]:H[p]+delta]
            S[np.mod(p,2):n:2,(p>1):n:2]= agrupar(tmp, f//2)
        else:
            # otros cuadrantes
            S[np.mod(p,2):n:2,(p>1):n:2]= A[ V[p]:V[p]+delta, H[p]:H[p]+delta ] 
    
    return S

def noise_estimation(A):
    n= A.shape[0]
    p=3
    tmp = A[np.mod(p,2):n:2,(p>1):n:2]    
    #tmp = np.zeros_like(tmp)  #que pasa si borramos un cuadrante
    sigma_noise = np.median(np.abs(tmp.flatten()))/0.6745

    return sigma_noise  

def  desagrupar(A, m, nse):
    n= A.shape[0]
    S = np.zeros([n,n])
    delta = n//2

    V,H= cuadrantes(delta)
    #haar decode
    for p in range(0,4):
        # Posiciones de la imagen       
        tmp = A[np.mod(p,2):n:2,(p>1):n:2]
        
        #caracteristicas    
        if (p==0 and m>2):
            #cuadrante PPAL
            S[ V[p]:V[p]+delta, H[p]:H[p]+delta]= desagrupar(tmp, m//2, nse)
        else:
            # otros cuadrantes
            #INSERTAR EL FILTRO ACÁ
            if (p==0 and m%2==0):
                S[ V[p]:V[p]+delta, H[p]:H[p]+delta ] = tmp
            else:
                y = np.maximum(np.abs(tmp)-nse,0)
                denoising = y/(y+nse)* tmp
                S[ V[p]:V[p]+delta, H[p]:H[p]+delta ] = denoising  
    
    return S

def cuadrantes(sz):
    #limites de cuadrantes
    V=np.zeros([2,2])
    for i in range(0,2):
        for j in range(0,2):
            ini  = sz*(np.mod(i,2))
            V[i,j] = ini
    
    H= V
    V= np.transpose(V)
    V= V.flatten().astype(int)
    H= H.flatten().astype(int)
    return (V, H)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# programa ppal
img = cv2.imread('imagenes/cameraman.png', cv2.IMREAD_GRAYSCALE)
img = cv2.normalize(img.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
noise = np.random.random(img.shape)*0.1
img = img + noise


iter = 7      #numero de subdivisiones (máx 8 niveles)


m = 2**iter  #niveles

H = haarmtx(m)
enc = encode_wavelet(img, H, 0)

nse = noise_estimation(enc)
print("Ruido estimado: {:3.8f}".format(nse))

grp =  desagrupar(enc,m, nse)
dec = agrupar(grp,m)
out = encode_wavelet(dec, H, 1)

#resultados
fig = plt.figure(figsize=(10,5))
plt.subplot(121)
plt.imshow(img, cmap= cm.gray)
plt.axis('off')
plt.title('Imagen Original')

plt.subplot(122)
plt.imshow(out, cmap= cm.gray)
plt.axis('off')
plt.title('Imagen Filtrada')
plt.show()

fig = plt.figure()
plt.subplot(121)
plt.imshow(enc, cmap= cm.gray)
plt.axis('off')
plt.title('Codificación de Haar')

plt.subplot(122)
plt.imshow(grp, cmap= cm.gray)
plt.axis('off')
plt.title('Agrupación en  niveles')
plt.show()
