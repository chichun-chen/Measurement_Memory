import numpy as np


def GF2_Gauss_elimination(A):
    n, m = A.shape

    def swap_row(i,j):
        temp = np.copy(A[i])
        A[i] = A[j]
        A[j] = temp

    i = 0
    j = 0
    while i < n and j < m:
        # Find the ith pivot
        i_max = np.argmax(A[i:, j]%2) + i
        if A[i_max, j]%2 == 0 :
            j += 1
        else:
            swap_row(i, i_max)
            # Gauss elimination
            for k in range(n):
                if A[k,j]%2 == 0 or k == i:
                        pass
                else:
                    for l in range(j, m):
                        A[k,l] = (A[k,l] + A[i,l])%2
            i += 1
            j += 1

    return A

def sympletic_subspace_encoding(Pauli:str):
    N1 = []
    N2 = []
    for w in Pauli:
        if w == 'X':
            N1.append(1)
            N2.append(0)
        elif w == 'Y':
            N1.append(1)
            N2.append(1)
        elif w == 'Z':
            N1.append(0)
            N2.append(1)
        else:
            N1.append(0)
            N2.append(0)
    return np.array(N1+N2)

def subspace_inner_product(a, b):
    N = len(a)//2
    A = np.array([[0,1],[1,0]])
    B = np.identity(N)
    J = np.kron(A,B)
    return np.dot(a, np.dot(J,b)) % 2

def decoding(M):
    Ps = []
    N = M.shape[1]//2
    for v in M:
        P = ''
        for i in range(N):
            if v[i] == 0:
                if v[i+N] == 0: 
                    P += 'I'
                else:
                    P += 'Z'
            else:
                if v[i+N] == 0: 
                    P += 'X'
                else:
                    P += 'Y'
        Ps.append(P)
    return Ps

def remove_zero(E):
    for i,v in reversed(list(enumerate(E))):
        if sum(v) == 0:
            E = np.delete(E, i, 0)
    return E




def Gram_Schmidt(X):
    n, dim = X.shape
    
    for i in range(n):
        for j in range(i):
            X[i] = X[i] - np.dot(X[i],X[j])/np.dot(X[j],X[j]) * X[j]
        X[i] = X[i]/np.sqrt(np.dot(X[i],X[i]))
    
    return X
            




