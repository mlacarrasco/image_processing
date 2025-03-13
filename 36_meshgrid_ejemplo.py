import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm

x  = np.linspace(-7,7,70)
y  = np.linspace(-7,7,70)
X,Y = np.meshgrid(x,y)
P   = X-Y

f   = np.cos(P)


fig = plt.figure()
ax  = fig.add_subplot(111,projection='3d')
surf = ax.plot_surface(X, Y, f, cmap=cm.rainbow,linewidth=0, antialiased=True)
plt.show()