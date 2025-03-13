import cv2
import codecs, json
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

import skimage
import numpy.matlib
from skimage.filters import threshold_otsu
from skimage import measure


img = cv2.imread('imagenes/rice.png')

m = img.shape[0]
n = img.shape[1]

N = 25

# MORPH_CROSS, MORPH_ELLIPSE, MORPH_RECT
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(N,N))

#apertura
tmp = cv2.erode(img,kernel)
apertura = cv2.dilate(tmp,kernel)
resta = cv2.subtract(img,apertura)

gray = resta[:,:,0]

# Otsu threshold		
thresh = threshold_otsu(gray)

#binary = operacion gray > thresh
(thresh, binary) = cv2.threshold(gray, thresh, 255, 0)

fig, axes = plt.subplots(ncols=2, figsize=(12, 6.5))
ax = axes.ravel()
ax[0] = plt.subplot(1, 2, 1)
ax[1] = plt.subplot(1, 2, 2, sharex=ax[0], sharey=ax[0])

ax[0].imshow(gray, cmap="gray")
ax[0].set_title('Imagen Original')
ax[1].imshow(binary, cmap= "gray")
ax[1].set_title('Binaria')

plt.show()


# componentes conectados
all_labels = measure.label(binary)
plt.figure()
plt.imshow(all_labels)
plt.title ('Componentes conectados')
plt.show()

# Analisis de las regiones
centroids = []
areas = []
newimg = np.zeros((m,n,3), dtype='uint8')

fig, ax = plt.subplots(figsize=(10, 6))
ax.imshow(binary, cmap="gray")

for region in measure.regionprops(label_image=all_labels):

    cx, cy = region.centroid[0], region.centroid[1]
    idx= region.image
    sls = region.coords
    
    areas.append(region.area)
    centroids.append((cx, cy))
    if region.area>200:
        
        plt.scatter(cy, cx, marker="x", color="red", s=20)
        minr, minc, maxr, maxc = region.bbox
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                  fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)
        
        newimg[sls[:,0], sls[:,1], 1]= 255


ax.imshow(newimg, cmap="gray")
plt.title('Regiones segmentadas')
plt.tight_layout()
plt.show()

# busqueda de contornos y simplificación
cnts, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
fig, ax = plt.subplots(figsize=(10, 6))
coordsmat = []
coords = []
ids  = 0
for c in cnts:
    # approximate the contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.015 * peri, True)
    #print(approx)
    idsmat = np.matlib.repmat(ids, len(approx) ,1)
    for val in approx:
        plt.scatter(val[0][0], val[0][1], marker="x", color="red", s=2)
    coords = np.vstack(np.dstack((idsmat, approx)))
    coords = np.vstack((coords, coords[0,:]))
    if (ids==0):
        coordsmat = coords
    else:
        coordsmat = np.concatenate((coordsmat, coords))
    ids +=1

ax.imshow(binary, cmap="gray")
ax.set_axis_off()
plt.tight_layout()
plt.show()

#almacenamos las coordenadas en archivo JSON
#print(coords)
pd.DataFrame(coordsmat, columns=['index','x','y']).to_json('data.json',orient='split')

plt.figure()
counts, bins = np.histogram(areas, bins=20)
plt.hist(bins[:-1], bins, weights=counts)

plt.show()
