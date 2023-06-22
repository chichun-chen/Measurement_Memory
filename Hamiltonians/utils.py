import os

def read_Ising_Hamiltonian(N:int, directory="Hamiltonians"):
    path = os.path.abspath(os.path.join(directory, "Ising_Hamiltonian_{}qubit.txt".format(N)))
    
    H = []
    group_ops = []
    group_coeffs = []
    with open(path) as f:
        qubits, terms, groups = f.readline()[:-1].split(' ')

        for _ in range(int(terms)):
            s = f.readline()[:-1].split(' ')
            group_ops.append(s[0])
            group_coeffs.append(float(s[1]))
        H.append((group_ops, group_coeffs))

    return H


def read_molecule_Hamiltonian(mol=None, grouping_type=None, directory="Hamiltonians"):
    path = os.path.abspath(os.path.join(directory, "{}_{}_Hamiltonian.txt".format(mol,grouping_type.upper())))

    def read_in(obj):
        # obj = 'Pauli' : read Pauli operators
        # obj = 'Givens' : read givens parameters
        # obj = 'Phase' : read phase
        obj = obj.lower()
        s = f.readline()[:-1].split(' ')
        if obj == 'pauli':
            group_ops.append(s[0])
            group_coeffs.append(float(s[1]))
        elif obj == 'givens':
            if float(s[0]) == 0 :
                pass
            else:
                group_givens.append((float(s[0]),
                                     int(s[1]),
                                     int(s[2])))
        elif obj == 'phase':
            group_phase.append(tuple(float(p) for p in s))

    def pack_up(obj, group_ops, group_coeffs, group_givens, group_phase):
        # Append a group to list
        obj = obj.lower()
        if obj == 'pauli':
            H.append((group_ops, group_coeffs))
            group_ops = []
            group_coeffs = []
        elif obj == 'givens':
            Givens.append(group_givens)
            group_givens = []
        elif obj == 'phase':
            Phase.append(group_phase)
            group_phase = []
        return group_ops, group_coeffs, group_givens, group_phase

    H = []
    Givens = []
    Phase = []
    group_ops = []
    group_coeffs = []
    group_givens = []
    group_phase = []
    if grouping_type.lower() == 'fg':
        objs = ['Phase', 'Givens', 'Pauli']
    else:
        objs = ['Pauli']

    with open(path) as f:
        qubits, terms, groups = f.readline()[:-1].split(' ')

        for _ in range(int(groups)):
            for _ in range(len(objs)):
               obj, items = f.readline()[:-1].split(' ')
               for item in range(int(items)):
                   read_in(obj)
               group_ops, group_coeffs, group_givens, group_phase=\
               pack_up(obj, group_ops, group_coeffs, group_givens, group_phase)
            # Skip blank line
            f.readline()

        T = sum([len(H[i][0]) for i in range(len(H))])
        assert T == int(terms) and len(H) == int(groups),\
               "Fail to read Hamiltonian correctly. Check the file format."
    if grouping_type.lower() == 'fg':
        return H, Givens, Phase
    else: 
        return H

