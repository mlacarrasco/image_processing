import cv2
import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt 

def filtro_mediana_adaptiva(A):
    m,n = A.shape
    Smax = 16
    ws   = 4
    J = np.zeros([m,n])

    for i in range(0,m-Smax):
        for j in range(0,n-Smax):
            sw     = 1
            Zxy    = A[i,j]
            wadapt = ws
            while(sw):   
                B = A[i:i+wadapt, j:j+wadapt]
                      
                pixel, newwin = bloque(B,Zxy,wadapt,Smax)
                if ( pixel != -1):
                    J[i,j]=pixel
                    sw = 0
                else:
                    wadapt=newwin
            
    return (np.uint8(J[0:m-Smax, 0:n-Smax]))
                
def bloque(A,Zxy, wadapt, Smax):

    newwin= wadapt

    Zmin= np.min(A.flatten())
    Zmax= np.max(A.flatten())
    Zmed= np.median(A.flatten())
    px = -1

    A1= Zmed - Zmin
    A2= Zmed - Zmax

    if (newwin<=Smax):
    
        #% Nivel A
        if (A1>0) & (A2<0):
            #%nivel B
            B1= float(Zxy)-float(Zmin)
            B2= float(Zxy)-float(Zmax)
        
            if (B1>0) & (B2<0):
                px=Zxy
            else:
                px=Zmed
            
        else:
            # Incrementamos el tamaño de 
            # la ventana
            newwin= newwin+1
    else:
        px=Zxy
    
    return(px, newwin)

#programa ppal
img = cv2.imread('imagenes/rice.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#Ruido impulsional
mat_noise=np.random.random(gray.shape)
noise_img = gray.copy()  #copiamos la matriz original
sp_noise_white= mat_noise>=0.7 #matriz logica
sp_noise_black= mat_noise<=0.3 #matriz logica

noise_img[sp_noise_white] =255
noise_img[sp_noise_black] = 0

cv2.imwrite('imagenes/imagen_ruido_2.png',noise_img)

#aplicamos el filtro mediana_adaptiva
filtro = filtro_mediana_adaptiva(noise_img)

plt.figure()
plt.imshow(noise_img, cmap="gray", vmin=0,vmax=255)
plt.show()

plt.figure()
plt.imshow(filtro, cmap="gray",vmin=0,vmax=255)
plt.show()


