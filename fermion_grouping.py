import numpy as np
from copy import deepcopy

def QR_decomposition(A):
    '''
     QR decomposition with givens rotation
     Input : matrix A
     Output : (Q, R, G), where A = QR, and G = [(theta,p,q), (theta,p,q),...]
              is a series of parameters of givens rotaion [g1, g2,...],
              where g1g2g3... = Q    
    '''    
    # Function for obtaining givens matrix elements
    def givens_matrix_entries(a, b):
        r = np.sqrt(a*a + b*b)
        s = b/r
        c = a/r
        return s, c

    def robust_arccos(s,c):
        if c > 0 :
            if s > 0: return np.arccos(c)
            else: return -np.arccos(c)
        else:
            if s > 0: return np.arccos(c)
            else: return -np.arccos(c)   
    
    rows, cols = A.shape
    Q = np.identity(rows)
    R = np.copy(A)
    Gs = []
    # Iterate through the lower triangle
    for j in range(cols-1):
        for i in range(rows-1, j, -1):
            if R[i,j] != 0:
                s, c = givens_matrix_entries(R[i-1,j], R[i,j])
                G = np.identity(rows)
                G[i-1,i-1] = c
                G[i,i] = c
                G[i,i-1] = s
                G[i-1,i] = -s
                # Givens matrix parameters (theta, p, q)
                Gs.append((robust_arccos(s,c), i-1, i))
                R = G.T @ R
                Q = Q @ G
    return np.round(Q,16), np.round(R,16), Gs

def To_anti_normal_order(h1, h2):
    # Szabo's physicists' notation -> chemists' notation
    N  = len(h1)
    T = np.zeros((N,N))
    V = np.zeros((N,N,N,N))
    for p in range(N):
        for q in range(N):
            E = 0
            for r in range(N):
                E += h2[p,r,r,q]
                for s in range(N):
                    V[p,q,r,s] = h2[p,s,q,r]
            T[p,q] = h1[p,q] - E
    return T, V

def rewrite_two_body_tensor(V):
    # Rewrite tensor index from openfermion to Szabo's physicists' notation (pqsr -> pqrs)
    N = len(V)
    V_new = deepcopy(V)
    for r in range(N):
        for s in range(N):
            V_new[:,:,r,s] = V[:,:,s,r]
    return np.round(V_new,16)

def check_normal_order_8fold_symmetry(V, index:list):
    # check 8-fold symmetry in Szabo's physicists' notation
    assert len(index) == 4, "Only support 4 dimension tensor"
    p,q,r,s = index
    precision = 14
    val = np.round(V[p,q,r,s],precision) 
    #print("Value : {}".format(val))
    if np.round(V[r,q,p,s],precision) != val: return False
    if np.round(V[p,s,r,q],precision) != val: return False
    if np.round(V[r,s,p,q],precision) != val: return False
    if np.round(V[q,p,s,r],precision) != val: return False
    if np.round(V[s,p,q,r],precision) != val: return False
    if np.round(V[q,r,s,p],precision) != val: return False
    if np.round(V[s,r,q,p],precision) != val: return False
    return True

def check_anti_normal_order_8fold_symmetry(V, index:list):
    # check 8-fold symmetry in chemists' notation
    assert len(index) == 4, "Only support 4 dimension tensor"
    p,q,r,s = index
    precision = 14
    val = np.round(V[p,q,r,s],precision) 
    #print("Value : {}".format(val))
    if np.round(V[q,p,r,s],precision) != val: return False
    if np.round(V[p,q,s,r],precision) != val: return False
    if np.round(V[q,p,s,r],precision) != val: return False
    if np.round(V[r,s,p,q],precision) != val: return False
    if np.round(V[s,r,p,q],precision) != val: return False
    if np.round(V[r,s,q,p],precision) != val: return False
    if np.round(V[s,r,q,p],precision) != val: return False
    return True
