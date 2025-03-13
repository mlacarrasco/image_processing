import cv2
import matplotlib.pyplot as plt
import numpy as np

#lectura de la imagen
img=  cv2.imread('galaxia.jpg', cv2.IMREAD_GRAYSCALE)

#dimensiones
n,m = img.shape

gray = img

#ciclo for
for i in range(2):
    #ruido aleatorio normal
    noise= np.random.random((n,m))

    #normalizamos el ruido -1 a 1
    noise = cv2.normalize(noise.astype('float'), None, -1.0, 1.0, cv2.NORM_MINMAX)
    gray= noise+gray


#normalizamos
avg=  np.divide(gray,np.sum(gray))
out = cv2.normalize(avg.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)

plt.imshow(out, cmap= "gray")
plt.show()

