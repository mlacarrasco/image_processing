import cv2
import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt 

def filtro_ruido_local(A):

    var_N = 0.0008   #varianza estimada del ruido 
    B = A.flatten()
    n = len(B)
    var_L = np.var(B) #varianza en la mascara
    
    mu    = np.mean(B)
    g     = B[np.uint8(n/2)]     
    f     =  g - (var_N/var_L)*(g-mu)
    return f   
        
#programa ppal.
img = cv2.imread('imagenes/cameraman.png') #imagen ideal f(x,y).


gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray_norm = cv2.normalize(gray.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)

#generar ruido aleatorio
noise = np.random.random(gray.shape)*0.10
noise_img = gray_norm + noise  #imagen real g(x,y)

#seccion de una parte de la imagen
noise_img_roi = np.uint8(noise_img[25:110,85:180]*255)
cv2.imwrite('imagenes/imagen_ruido.png',noise_img_roi)


#aplicamos el filtro ruido local
filtro = ndi.generic_filter(noise_img, filtro_ruido_local, [5,5])


#estimar el cambio
dif= filtro-noise_img
delta= np.sum(dif.flatten())
print(delta)

fig, ax = plt.subplots(1,2)
fig.set_size_inches((8,6))
ax[0].imshow(noise_img[25:110,85:180], cmap="gray")

ax[0].set_title('Imagen original')
ax[1].imshow(filtro[25:110,85:180], cmap="gray")
ax[1].set_title('Imagen filtrada')
plt.show()




