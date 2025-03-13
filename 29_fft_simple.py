import numpy as np
from cmath import exp, pi
 
def fft(x):
    N = len(x)

    if N <= 1: 
        return x
    
    tmp = np.zeros(N, dtype=np.complex_)
    X = np.zeros(N, dtype=np.complex_)

    for r in range(0,N):    
        for n in range(0,N):
            tmp[n]= x[n]*exp(-2j*pi*r*n/N)    
        X[r]= np.sum(tmp)
    
    return X

def ifft(X):
    N = len(X)

    if N <= 1: 
        return X
    
    tmp = np.zeros(N, dtype=np.complex_)
    x   = np.zeros(N, dtype=np.complex_)

    for n in range(0,N):    
        for r in range(0,N):
            tmp[r]= X[r]*exp(2j*pi*r*n/N)  
        x[n]=1/N* np.sum(tmp)
    
    return x


A= [5.0, 4.0, 2.0, 1.0, 7.0, 9.0]
F= fft(A)
iF = ifft(F)
print(abs(iF))