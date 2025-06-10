import cv2
import matplotlib.pyplot as plt
import numpy as np

binaria = [[0,0,0,1],
           [0,1,1,0],
           [0,1,1,0],
           [0,0,0,1]]
binaria =  np.array(binaria, dtype='uint8')

binaria = cv2.resize(binaria, (100,100), cv2.INTER_AREA)*255
resultado = cv2.equalizeHist(binaria)


fig, ax = plt.subplots(nrows=2,ncols=2)
fig.set_size_inches(8,7)
ax[0,0].imshow(binaria, cmap='gray')
ax[0,0].set_title('imagen binaria')
ax[1,0].hist(binaria.flatten())
ax[1,0].set_title('Histograma binaria')

ax[0,1].imshow(resultado, cmap='gray')
ax[0,1].set_title('imagen ecualizada')
ax[1,1].hist(resultado.flatten())
ax[1,1].set_title('Histograma ecualizada')
plt.show()



