import os

def read_Hamiltonian(Hamiltonian:str, N=None, mol=None, grouping_type=None):
    
    if Hamiltonian.lower() == "ising":
        if N == None: raise Exception("qubit number N is not defined.")
        path = os.path.abspath(os.path.join("Hamiltonians", "Ising_Hamiltonian_{}qubit.txt".format(N)))
        #path = os.path.abspath(os.path.join(".", "Ising_Hamiltonian_{}qubit.txt".format(N)))
    elif Hamiltonian.lower() == "molecular":
        if mol == None: raise Exception("molecule is not defined.")
        if grouping_type == None: raise Exception("grouping_type is not defined (QWC or GC).")
        path = os.path.abspath(os.path.join("Hamiltonians", "{}_{}_Hamiltonian.txt".format(mol, grouping_type)))
        #path = os.path.abspath(os.path.join(".", "{}_{}_Hamiltonian.txt".format(mol, grouping_type)))
    Hg = []
    #Groups = []
    with open(path) as f:
        qubits, terms, groups = f.readline()[:-1].split(' ')
        
        group_ops = []
        group_coeffs = []
        T = 0
        G = 0
        for line in f.readlines():
            s = line.split(' ')
            if s[0] == '\n' or s[0] == None:
                Hg.append((group_ops, group_coeffs))
                #Hg.append(qml.Hamiltonian(group_coeffs, str_to_Pauli(group_ops), grouping_type='commuting'))
                group_ops = []
                group_coeffs = []
                G += 1
            else:
                group_ops.append(s[0])
                if s[1][-1] == '\n': s[1] = s[1][:-1]
                group_coeffs.append(float(s[1]))
                T += 1
        assert T == int(terms) and G == int(groups),\
               "Fail to read Hamiltonian correctly. Check if the last group end with '\n'."
    return Hg
