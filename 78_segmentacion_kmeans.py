import cv2
import numpy as np 
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
from skimage import filters

def mouse_clic(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONUP:
        src = param.copy()
        cv2.floodFill(src, None, (x, y), (255))
        cv2.imshow('Fill zone', src)
    
def filtro_mean(A):
    S= np.mean(A.ravel())    
    return S

def filtro_std(A):
    S= np.std(A.ravel())    
    return S

#programa ppal
winname = 'Rombo'
img  = cv2.imread('imagenes/rombo.png')
gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
rbo  = filters.roberts(gris)

std_f= ndi.generic_filter(rbo,filtro_std, [3,3])
std_m= ndi.generic_filter(rbo,filtro_mean, [3,3])

#matriz de propiedades de pixeles
subdata= np.c_[std_f.ravel() , std_m.ravel()]
#subdata = np.transpose(subdata)

#Estimación de clusters con kmeans
km = KMeans(n_clusters=2)
km = km.fit(subdata)

#grafico de propiedades
plt.figure()
plt.scatter(np.array(subdata[:,0]), np.array(subdata[:,1]),c=km.labels_, s=50, alpha=0.5)
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], c='red', s=50)
plt.show()

#buscamos los valores de la clase 1
bit_1= km.labels_==1
OP = np.zeros([gris.shape[0]*gris.shape[1]])
OP[bit_1]=255 #asignamos los valores a 255 (blanco)
OP =  np.uint8(np.reshape(OP, (gris.shape[0], gris.shape[1])))

cv2.namedWindow(winname)
cv2.setMouseCallback(winname, mouse_clic, OP)

while(1):
    cv2.imshow(winname, OP)
    if cv2.waitKey(20) & 0xFF ==ord('q'):
        break

cv2.destroyAllWindows()