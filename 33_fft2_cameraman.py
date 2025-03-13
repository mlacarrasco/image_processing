import cv2
import numpy as np

img = cv2.imread('cameraman.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

F = np.fft.fft2(gray)
fshift = np.fft.fftshift(F)
spectrum = np.log(np.abs(fshift))
out = cv2.normalize(spectrum, None, 0.0, 1.0, cv2.NORM_MINMAX)

plt.imshow(out,cmap="gray")
plt.show()