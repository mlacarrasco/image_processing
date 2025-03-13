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
    
    tmp = np.zeros(N, dtype=np.complex)
    x   = np.zeros(N, dtype=np.complex)

    for n in range(0,N):    
        for r in range(0,N):
            tmp[r]= X[r]*exp(2j*pi*r*n/N)  
        x[n]=1/N* np.sum(tmp)
    
    return x

A= np.array([[ 5.0, 9.0, 10.0],
    [ 8.0, 6.0, 9.0],
    [ 8.0 ,0.0, 2.0]])

F  = np.zeros([3,3], dtype=complex)
FF = np.zeros([3,3], dtype=complex)

for i in range(0,3):
    F[:,i]= fft(A[:,i])
   
for i in range(0,3):
    FF[i,:]= fft(F[i,:])

F2= np.fft.fft2(A)

print(FF)
print(F2)