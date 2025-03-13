import numpy as np
import matplotlib.pyplot as plt
fc=40  # Frecuencia de corte
N = 3   # orden del filtro


f= np.linspace(0, 2*fc, 2*fc)
H= 1/ (1+(np.power(f/fc,2*N)))

plt.plot(H)
plt.show()
