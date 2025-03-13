import cv2 
import numpy as np

#lectura de imagen
img = cv2.imread('imagenes/cameraman.png')
img = img[:,:,0]  #ocupamos en canal 0

#ciclo por cada valor binario
for k in range(0, 8):
  plane = np.full((img.shape[0], img.shape[1]), 2 ** k, np.uint8)
  res = plane & img
  x = res*255
  
  plt.imshow(x, cmap="gray")
  plt.show()
