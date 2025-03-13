import cv2
import matplotlib.pyplot as plt
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.draw import circle_perimeter
from skimage import data, color
import numpy as np
from matplotlib import cm

#programa ppal
gray = cv2.imread('imagenes/coins.jpg', cv2.IMREAD_GRAYSCALE)
edges = cv2.Canny(gray,200,255,apertureSize = 3)

plt.figure()
plt.imshow(edges, cmap='gray')
plt.show()


# Generamos un rango de radios de búsqueda
hough_radii = np.arange(30, 40, 2)
hough_res = hough_circle(edges, hough_radii)

# Seleccionamos solo los tres peaks
accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, total_num_peaks=5)

# Draw them
plt.figure(figsize=(10, 4))
image = color.gray2rgb(gray)
for center_y, center_x, radius in zip(cy, cx, radii):
    circy, circx = circle_perimeter(center_y, center_x, radius, shape=image.shape)
    image[circy, circx] = (220, 20, 20)

plt.imshow(image, cmap='gray')
plt.show()

