# This code is created by Chi Chun Chen @ Physics division, NCTS, Taipei, Taiwan   
# Execute the experiment by:                                                       
# python Experiment_{Type}.py {-MM} {output file #}                                
####################################################################################
import os
import sys
import time
import pennylane as qml
from pennylane import numpy as np
from measurement_memory import *
from grouping import *


def read_Hamiltonian(mol:str, grouping_type:str):
    path = os.path.abspath(os.path.join("Hamiltonians", "{}_{}_Hamiltonian.txt".format(mol, grouping_type)))
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


#def check_big_groups(Hg:list, n:int):
#    big_groups = []
#    for i,h in enumerate(Hg):
#        if len(h.ops) >= n:
#            big_groups.append(i)
#    return big_groups

p = 1
def ansatz(params, qubits,  depth=p):
    for q in range(qubits):
        qml.U3(params[3*q],params[3*q+1],params[3*q+2], wires=q)
    for d in range(1,depth+1):
        for q in range(qubits-1):
            qml.CNOT(wires=[q,q+1])
        for q in range(qubits):
            qml.U3(params[3*(d*qubits+q)], params[3*(d*qubits+q)+1], params[3*(d*qubits+q)+2], wires=q)

def measurement_rotation(T, Q):
    # Input: Original lagrangian basis and the new single qubit basis
    # Do : Rotate to single measurement basis
    if T == None :
        single_qubit_basis_rotation(is_QWC(Q, return_basis=True))
    else:
        assert len(T) == len(Q), "Two basis has differnt dimensions."
        q = len(T[0])

        for i in range(len(T)):
            qml.PauliRot(-np.pi/2, Q[i], range(q))
            qml.PauliRot(-np.pi/2, T[i], range(q))
            qml.PauliRot(-np.pi/2, Q[i], range(q))
        single_qubit_basis_rotation(is_QWC(Q, return_basis=True))


def cost(x):
    E = 0
    for i,h in enumerate(Hg):
        T, Q = basis[i]
        E += evaluate_eigenstate_MM(sample_circuit(x, T, Q),\
                                    h, memory=M[i], memory_states=5000)
    return E

def cost0(x):
    E = 0
    for i,h in enumerate(Hg):
        T, Q = basis[i]
        E += evaluate_eigenstate(sample_circuit(x, T, Q), h)
    return E


def read_initial_parameters():
    try:
        initial_params = []
        with open("Mol_initial_parameters.txt", 'r') as f:
            for line in f.readlines():
                s = line.split(' ')
                if s[-1] == '\n' : s = s[:-1]
                for i,n in enumerate(s):
                    s[i] = float(n)
                initial_params.append(s)
    except:
        raise FileExistsError("Must generate initial parameters.")
    return initial_params

def get_file_name(is_MM, grouping_type):
    if is_MM :
        return "Molecular_{}_MM{}.txt".format(grouping_type,num_exp)
    else:
        return "Molecular_{}_normal{}.txt".format(grouping_type,num_exp)

def write_results(path, mol, N, time, overwrite=True):
    if overwrite:
        f = open(path, 'w') 
        f.write("Mol N Time\n")
    else:
        f = open(path, 'a')

    f.write("{} {} {}\n".format(mol, N, time))
    f.close()

if __name__ == '__main__':
    argList = sys.argv[1:]
    # if used MM
    is_MM = False
    if  "-MM" in argList:
        is_MM = True
        argList.remove("-MM")
    # Number of experiment
    num_exp = int(argList[0])   
    ##############
    # Parameters #
    ##############
    moleculars = ["H2", "H4", "LiH", "H2O"]
    qubits = [4, 8, 12, 14]
    grouping_type = 'QWC'   # 'GC': general commuting, 'QWC': qubit-wise commuting
    max_itr = 100
    gradient_method = 'parameter_shift'
    # Read initial parameters
    init_params = read_initial_parameters()[num_exp-1]
    ##############################################################################  

    for i, mol in enumerate(moleculars):
        H = read_Hamiltonian(mol, grouping_type)
        N = qubits[i]
        # Get groups with # of ops >= N
        ## big_groups = check_big_groups(Hg, N)
        # Get measurement information for each group
        Hg, basis = basis_transformation(H)
            
        dev = qml.device("lightning.qubit", wires=N, shots=100*N)
        @qml.qnode(dev)
        def sample_circuit(params, T, Q):
            ansatz(params, N)
            measurement_rotation(T, Q)
            return qml.counts()


        params = init_params[:3*(p+1)*N]
        if is_MM :
            start = time.process_time()
            M = [{} for _ in range(len(Hg))]
            for itr in range(max_itr):
                params = GradientDescent(cost, params, gradient_method, learning_rate=0.05)
                obj_value = cost(params)
                #print(obj_value)
            end = time.process_time()
        else:
            start = time.process_time()
            for itr in range(max_itr):
                params = GradientDescent(cost0, params, gradient_method, learning_rate=0.05)
                obj_value = cost0(params)
                #print(obj_value)   
            end = time.process_time()
    
        time_used = np.round(end-start, 3)
        
        path = get_file_name(is_MM, grouping_type)
        overwrite = False
        if i == 0 : overwrite = True
        write_results(path, mol, N, time_used, overwrite)
        
