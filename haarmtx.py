"""
% Universidad Adolfo Ibañez
% Facultad de Ingeniería y Ciencias

% Procesamiento Digital de Imágenes
% Miguel Carrasco (miguel.carrasco@uai.cl)
% rev.$1.2.$ date: 09/05/2020
"""

import numpy as np 

def haarmtx(n):
    #basado en HAARMTX Compute Haar orthogonal transform matrix.
    # de Frederic Chanal (f.j.chanal@student.tue.nl) - 2004
    H= np.zeros([n,n])
    a=1/np.sqrt(n)
    for i in range(0,n):
        H[0,i] = a

    for k in range(1,n):
        p = int(np.log2(k))
        q = k-2**p+1
        t1 = n/2**p
        sup = int(q*t1)
        mid = int(sup-t1/2)
        inf = int(sup-t1)
        t2=2**(p/2)*a
        for j in range(0,inf):
            H[k,j]=0
    
        for j in range(inf,mid):
            H[k,j]=t2
        
        for j in range(mid,sup):
            H[k,j]= -t2
    
        for j in range(sup,n):
            H[k,j]=0
    
    return (H)