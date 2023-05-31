# This code is created by Chi Chun Chen @ Physics division, NCTS, Taipei, Taiwan   
# Execute the experiment by:                                                       
# python Experiment_{Type}.py {-MM} {output file #}                                
####################################################################################
import os
import sys
import time
import pennylane as qml
from pennylane import numpy as np
from Hamiltonians.utils import read_Hamiltonian
from measurement_memory import *
from grouping import *



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

def single_qubit_basis_rotation(M:list):
    # Input : Measurement Pauli basis
    # Do :  Single qubit rotation to measurement basis
    for i,m in enumerate(M):
        if m == 'X':
            qml.Hadamard(i)
        elif m == 'Y':
            qml.adjoint(qml.S(i))
            qml.Hadamard(i)
        elif m == 'Z' or m == 'I':
            pass
        else:
            raise Exception("Did not receive single qubit Pauli basis.")

def cost_MM(x):
    # cost function for measurement memory
    E = 0
    for i,g in enumerate(Hg):
        T, Q = basis[i]
        E += evaluate_eigenstate_MM(sample_circuit(x, T, Q),\
                                    g, memory=M[i])
    return E

def cost0(x):
    # cost function for normal evaluation
    E = 0
    for i,g in enumerate(Hg):
        T, Q = basis[i]
        E += evaluate_eigenstate(sample_circuit(x, T, Q), g)
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
    grouping_type = 'GC'   # 'GC': general commuting, 'QWC': qubit-wise commuting
    max_itr = 100
    gradient_method = 'parameter_shift'
   
     # Read initial parameters
    init_params = read_initial_parameters()[num_exp-1]  

    for i, mol in enumerate(moleculars):
        H = read_Hamiltonian("Molecular", mol=mol, grouping_type=grouping_type)
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
                params = GradientDescent(cost_MM, params, gradient_method, learning_rate=0.05)
                obj_value = cost_MM(params)
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
        
