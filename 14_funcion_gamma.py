import cv2
import numpy as np
import matplotlib.pyplot as plt

def gamma_correction(img, factor):
	img = img/255.0
	img = cv2.pow(img, factor)
	return np.uint8(img*255)

#*******************
# programa principal
#*******************

#lectura imagen
img = cv2.imread('imagenes/cameraman.png')

#llamado a funcion
output = gamma_correction(img, 5)

plt.imshow(output)
plt.show()
