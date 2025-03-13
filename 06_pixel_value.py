import cv2

#lectura de imagen
img = cv2.imread('imagenes/lena.png')

#acceso a nivel de pixel
pixel = img[265,327,:]
print(pixel)
