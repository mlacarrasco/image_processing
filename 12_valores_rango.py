import cv2
import matplotlib.pyplot as plt

#lectura imagen
img = cv2.imread('imagenes/cameraman.png')

#accesp a posiciones de la imagen
print(img[0:5,0:5,0])
