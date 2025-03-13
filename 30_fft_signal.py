import numpy as np
import matplotlib.pyplot as plt

t= np.linspace(0,1, num=1000)
y = 0.8*np.sin(2*np.pi*t*10)

plt.plot(t, y)
plt.show()

F = np.fft.fft(y)
N= len(F)

p= np.linspace(0,49,dtype=int, num=50)
plt.bar(x=p, height=abs(F[p])/N*2)
plt.show()

