import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


img = cv2.imread('imagenes/cameraman.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray_norm = cv2.normalize(gray.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)

noise = np.random.normal(0.2, 0.05, gray.shape)
noise_out = gray_norm + noise


plt.figure()
plt.imshow(noise_out, cmap="gray")
plt.show()
