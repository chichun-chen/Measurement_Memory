import numpy as np


def basis_transformation(H:list):
    Hg = []
    basis = []
    for i in range(len(H)):
        Hg_temp, T, Q = get_measurement_basis(H[i])
        Hg.append(Hg_temp)
        basis.append((T, Q))
    return Hg, basis

def get_measurement_basis(H:list):
    # Input : Mutually commuting Hamiltonian group H
    # Output : QWC_Hamiltonian H'=U*HU & corresponding U
    G = H[0] 
    old_coeffs = H[1]
    if not is_commuting(G):
        raise Exception('Hamiltonian given are not mutually commuting.')
    if is_QWC(G):
        return H, None, G

    # Encode to sympletic vector space over GF(2) 
    B = []
    for P in G:
        B.append(sympletic_subspace_encoding(P))
    G_b = np.array(B)
    # Compute orthogonal basis (Q & T)
    lagrangian_basis = get_lagrangian_subspace(G_b)
    new_basis = get_single_qubit_basis(lagrangian_basis)
    T = [decode_to_Pauli(s) for s in lagrangian_basis ]
    Q = [decode_to_Pauli(s) for s in new_basis ]

    new_pauli_vectors, phase = basis_transform(G_b, lagrangian_basis, new_basis)
    new_coeffs = [old_coeffs[i]*phase[i] for i in range(len(old_coeffs))]
    new_G = [decode_to_Pauli(s) for s in new_pauli_vectors]

    new_H = (new_G, new_coeffs)
    return new_H, T, Q


def is_commuting(P:list):
    # Input : List of Pauli obsevables
    # OutPut : True/False (mutually commuting or not)
    for i in range(len(P)):
        for j in range(i+1, len(P)):
            x = 0
            for q in range(len(P[0])):
                if P[i][q] != 'I' and P[j][q] != 'I':
                    if P[i][q] != P[j][q]: 
                        x += 1
            if x%2 != 0 : return False
    return True

def is_QWC(P:list, return_basis=False):
    # Input : List of Pauli obsevables
    # OutPut : True/False (qubit-wise commuting or not)
    q = len(P[0])
    B = ['I']*q
    for i in range(q):
        for obs in P:
            if obs[i] != 'I':
                if B[i] == 'I': B[i] = obs[i]
                if B[i] != obs[i]: return False
    if return_basis: 
        return B
    return True
                

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


def decode_to_Pauli(v):
    N = len(v)//2
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
    return P

def remove_zero(E):
    for i,v in reversed(list(enumerate(E))):
        if sum(v) == 0:
            E = np.delete(E, i, 0)
    return E


def binary_null_space(binary_matrix):
    '''
    Return the basis of binary null space of the given binary matrix. 

    Return: a list of binary vectors that forms the null space of the given binary matrix.
    Modify: gauss_eliminated binary_matrix. 
    '''
    dim = len(binary_matrix[0, :])
    I = np.identity(dim)

    # Apply Gauss Elimination on [A; I]
    for i in range(dim):
        non_zero_row = np.where(binary_matrix[:, i] == 1)

        # if exists a non zero index
        if len(non_zero_row[0]) != 0:
            non_zero_row = non_zero_row[0][0]
            for j in range(i + 1, dim):
                if binary_matrix[non_zero_row, j] != 0:
                    binary_matrix[:, j] = (binary_matrix[:, i] +
                                           binary_matrix[:, j]) % 2
                    I[:, j] = (I[:, i] + I[:, j]) % 2

    null_basis = []
    # Find zero column index
    for i in range(dim):
        if all(binary_matrix[:, i] == 0):
            null_basis.append(I[:, i])
    return null_basis

def get_lagrangian_subspace(binary_matrix):
    '''
    Given a list of vectors. 
    Find the lagrangian subspace that spans the vectors. 

    Return: A list of vectors that forms the lagrangian subspace and
    spans the space of the given matrix. 
    '''
    null_basis = binary_null_space(np.array(binary_matrix))
    lagrangian_basis = binary_symplectic_gram_schmidt(null_basis)
    for vec in lagrangian_basis:
        flip_first_second_half(vec)

    return lagrangian_basis


def flip_first_second_half(vector):
    '''
    Modify: Flip the first half the vector to second half, and vice versa. 
    '''
    dim = len(vector) // 2
    tmp = vector[:dim].copy()
    vector[:dim] = vector[dim:]
    vector[dim:] = tmp

def binary_symplectic_gram_schmidt(coisotropic_space):
    '''
    Accepts a list of binary vectors, basis which
    forms coisotropic space in 2*dim binary symplectic space (so len of list > dim)
    Use Symplectic Gram-Schmidt to find the lagrangian subspace within the coisotropic space
    that seperates the binary space into X, Y subspaces such that <x_i, y_i> = 1. 

    Return: A list of the basis vector of the lagrangian subspace X. 
    Modify: coisotropic space to isotropic space. 
    '''
    dim = len(coisotropic_space[0]) // 2
    symplectic_dim = len(coisotropic_space) - dim

    x_set = []
    # If is lagrangian already
    if symplectic_dim == 0:
        x_set = coisotropic_space
    else:
        for i in range(symplectic_dim):
            x_cur, y_cur = pick_symplectic_pair(coisotropic_space)
            binary_symplectic_orthogonalization(coisotropic_space, x_cur, y_cur)
            x_set.append(x_cur)
        x_set.extend(coisotropic_space)
    return x_set

def pick_symplectic_pair(coisotropic_space):
    '''
    Pick out one pair in the symplectic space such that 
    <sympX, sympY> = 1

    Return: A symplectic pair. 
    Modify: Pops the chosen pair from given list of vectors. 
    '''
    for i, ivec in enumerate(coisotropic_space):
        for j, jvec in enumerate(coisotropic_space):
            if i < j:
                if binary_symplectic_inner_product(ivec, jvec) == 1:
                    y = coisotropic_space.pop(j)
                    x = coisotropic_space.pop(i)
                    return x, y
            

def binary_symplectic_orthogonalization(space, x, y):
    '''
    Symplectically orthogonalize all vectors in space
    to the symplectic pair x and y

    Modify: The linear space of space is modified such that each basis vector 
    is symplectically orthogonal to x and y. 
    '''
    for i in range(len(space)):
        vec = space[i]
        x_overlap = binary_symplectic_inner_product(vec, x)
        y_overlap = binary_symplectic_inner_product(vec, y)
        vec = vec + y_overlap * x + x_overlap * y
        space[i] = vec % 2


def binary_symplectic_inner_product(a, b):
    '''
    Return the binary symplectic inner product between two binary vectors a and b. 

    Return: 0 or 1. 
    '''
    if not len(a) == len(b):
        raise TequilaException(
            'Two binary vectors given do not share same number of qubits. ')
    dim = len(a) // 2
    re = a[:dim] @ b[dim:] + b[:dim] @ a[dim:]

    return re % 2

def get_single_qubit_basis(lagrangian_basis):
    '''
    Find the single_qubit_basis such that single_qubit_basis[i] anti-commutes
    with lagrangian_basis[i], and commute for all other cases. 
    '''
    dim = len(lagrangian_basis)

    # Free Qubits
    free_qub = [qub for qub in range(dim)]
    pair = []

    for i in range(dim):
        cur_pair = find_single_qubit_pair(lagrangian_basis[i], free_qub)
        for j in range(dim):
            if i != j:
                if binary_symplectic_inner_product(cur_pair, lagrangian_basis[j] == 1):
                    lagrangian_basis[j] = (lagrangian_basis[i] +
                                           lagrangian_basis[j]) % 2
        pair.append(cur_pair)
    return pair

def find_single_qubit_pair(cur_basis, free_qub):
    '''
    Find the single qubit pair that anti-commute with cur_basis such that the single qubit is in free_qub 

    Return: Binary vectors representing the single qubit pair
    Modify: Pops the qubit used from free_qub
    '''
    dim = len(cur_basis) // 2
    for idx, qub in enumerate(free_qub):
        for term in range(3):
            pair = gen_single_qubit_term(dim, qub, term)
            # if anticommute
            if (binary_symplectic_inner_product(pair, cur_basis) == 1):
                free_qub.pop(idx)
                return pair

def gen_single_qubit_term(dim, qub, term):
    '''
    Generate single qubit term on the given qubit with given term (0, 1, 2 represents z, x, y) 

    Return: A binary vector representing the single qubit term specified. 
    '''
    word = np.zeros(dim * 2)
    if term == 0:
        word[qub + dim] = 1
    elif term == 1:
        word[qub] = 1
    elif term == 2:
        word[qub] = 1
        word[qub + dim] = 1
    return word





def basis_transform(G, old, new):
    '''
    Transform a group of commuting Hamiltonian from old to new binary basis.

    Return: Pauli string in the new binary basis.
    '''
    n_qubit = len(G[0])//2
    new_pauli_vectors = []
    phases = []
    for P in G:
        old_basis_coeff = binary_solve(old, P)
        original_pauli_vec = np.zeros(n_qubit * 2)
        new_pauli_vec = np.zeros(n_qubit * 2)
        phase = 1
        for i, i_coeff in enumerate(old_basis_coeff):
            if i_coeff == 1:
                phase *= binary_phase(original_pauli_vec, old[i], n_qubit)
                original_pauli_vec = (original_pauli_vec + old[i]) % 2
                new_pauli_vec = (new_pauli_vec + new[i]) % 2
        new_pauli_vectors.append(new_pauli_vec)
        phases.append(1.0/phase)
    return new_pauli_vectors, phases

def binary_solve(basis, target):
    '''
    Get the expansion of the target in the given basis in binary space. 
    '''
    coeff = np.zeros(len(basis))
    tsf_mat, pivot = binary_reduced_row_echelon(basis)
    for i, pivot_idx in enumerate(pivot):
        if target[int(pivot_idx)] == 1:
            coeff = (coeff + tsf_mat[:, i]) % 2
    return coeff

def binary_reduced_row_echelon(basis):
    '''
    Get a list of basis vectors. 
    Perfrom reduced row echelon and return the pivot and the transformation matrix such that
    np.array(basis) @ transformation_matrix = reduced_row_echelon_form
    '''
    num_basis = len(basis)
    dim = len(basis[0])

    # Initiate. No change.
    tsf_mat = np.identity(num_basis)
    reduced_basis = [vec.copy() for vec in basis]
    pivot = np.zeros(num_basis)

    for i, i_col in enumerate(reduced_basis):
        non_zero_row = np.where(i_col == 1)[0][0]
        pivot[i] = non_zero_row
        for j, j_col in enumerate(reduced_basis):
            if (i != j and j_col[non_zero_row] == 1):
                reduced_basis[j] = (j_col + i_col) % 2
                tsf_mat[:, j] = (tsf_mat[:, i] + tsf_mat[:, j]) % 2
    return tsf_mat, pivot

def binary_phase(self_binary, other_binary, n_qubit):
    '''
    Obtain the phase due to binary pauli string self * other. Get 0, 1, 2, 3 for 1, i, -1, -i.
    '''
    def get_phase_helper(this, other):
        '''
        Return the phase incured due to multiplying this * other on a single qubit. 
        '''
        identity = [0, 0]
        x = [1, 0]
        y = [1, 1]
        z = [0, 1]
        if this == identity or other == identity or this == other:
            return 0
        elif this == x:
            if other == y:
                return 1
            else:
                return 3
        elif this == y:
            if other == z:
                return 1
            else:
                return 3
        elif this == z:
            if other == x:
                return 1
            else:
                return 3

    phase = 0
    for i in range(n_qubit):
        self_cur_qub = [self_binary[i], self_binary[i + n_qubit]]
        other_cur_qub = [other_binary[i], other_binary[i + n_qubit]]
        phase += get_phase_helper(self_cur_qub, other_cur_qub)
    phase = phase % 4

    if phase == 0:
        return 1
    elif phase == 1:
        return 1j
    elif phase == 2:
        return -1
    else:
        return -1j


