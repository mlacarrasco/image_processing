import cv2
import numpy as np 
from skimage import filters
from skimage import feature
import matplotlib.pyplot as plt

img = cv2.imread('cameraman.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_OTSU)
# Marker labelling
ret, markers = cv2.connectedComponents(thresh)
markers = cv2.watershed(img,markers)

watershed = np.zeros(gray.shape)
watershed[markers == -1] = [255]

roberts = filters.roberts(gray)
prewitt = filters.prewitt(gray)
sobel   = filters.sobel(gray)
sato =  filters.sato(gray)
canny = feature.canny(gray, sigma=2)

fig = plt.subplots(nrows=2, ncols=3, figsize=(8, 3))
ax1=plt.subplot(231)
ax2=plt.subplot(232)
ax3=plt.subplot(233)
ax4=plt.subplot(234)
ax5=plt.subplot(235)
ax6=plt.subplot(236)

ax1.imshow(roberts, cmap=plt.cm.gray)
ax1.axis('off')
ax1.set_title('Roberts')

ax2.imshow(prewitt, cmap=plt.cm.gray)
ax2.axis('off')
ax2.set_title('Prewitt')

ax3.imshow(sobel, cmap=plt.cm.gray)
ax3.axis('off')
ax3.set_title('Sobel')

ax4.imshow(sato, cmap=plt.cm.gray)
ax4.axis('off')
ax4.set_title('Sato')

ax5.imshow(canny, cmap=plt.cm.gray)
ax5.axis('off')
ax5.set_title(r'Canny $\sigma=3$')

ax6.imshow(watershed, cmap=plt.cm.gray)
ax6.axis('off')
ax6.set_title('Watershed')



plt.show()