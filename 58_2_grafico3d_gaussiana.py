import numpy as np
import matplotlib.pyplot as plt
from math import pi

t= 100
sigma = 20
ventana= np.linspace(-t/2, t/2, t)
u,v = np.meshgrid(ventana, ventana)
G= (1/(sigma**2*2*pi))*np.exp(-(u**2+v**2)/(2*sigma**2))
N= G/np.sum(G.flatten())  #normalizamos


fig = plt.figure()
ax  = plt.axes(projection='3d')
ax.plot_surface(u, v, N, cmap='Spectral')
plt.show()

fig = plt.figure()
ax  = plt.axes(projection='3d')
ax.contour3D(u, v, N, 50, cmap='binary')
plt.show()
