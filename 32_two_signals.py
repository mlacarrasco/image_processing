import numpy as np
import matplotlib.pyplot as plt

t= np.linspace(0,1,num =1000)  #tiempo
R = np.random.random(1000)

y = 0.5*np.sin(2*np.pi*25*t)+np.sin(2*np.pi*15*t)+0.2*R

F = np.fft.fft(y)
N = len(F)

p= np.linspace(1,49,dtype=int, num=50)
plt.bar(x=p, height=abs(F[p])/N*2)
plt.show()