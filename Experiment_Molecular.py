# This code is created by Chi Chun Chen @ Physics division, NCTS, Taipei, Taiwan   
# Execute the experiment by:                                                       
# python Experiment_{Type}.py {-MM} {output file #}                                
####################################################################################
import os
import sys
import time
import pennylane as qml
from pennylane import numpy as np
from measurement_memory import evaluate_eigenstate, evaluate_eigenstate_MM,\
                               GradientDescent, str_to_Pauli
from measurement_memory import get_measurement_list, measurement_rotation


def read_Hamiltonian(mol:str, grouping_type:str):
    path = os.path.abspath(os.path.join("Hamiltonians", "{}_{}_Hamiltonian.txt".format(mol, grouping_type)))
    Hg = []
    Groups = []
    with open(path) as f:
        qubits, terms, groups = f.readline()[:-1].split(' ')
        
        group_ops = []
        group_coeffs = []
        T = 0
        G = 0
        for line in f.readlines():
            s = line.split(' ')
            if s[0] == '\n' or s[0] == None:
                Groups.append(group_ops)
                Hg.append(qml.Hamiltonian(group_coeffs, str_to_Pauli(group_ops), grouping_type='commuting'))
                group_ops = []
                group_coeffs = []
                G += 1
            else:
                group_ops.append(s[0])
                if s[1][-1] == '\n': s[1] = s[1][:-1]
                group_coeffs.append(float(s[1]))
                T += 1
        assert T == int(terms) and G == int(groups), "Fail to read Hamiltonian correctly. Check if the last group end with '\n'."
    return Hg

def check_big_groups(Hg:list, n:int):
    big_groups = []
    for i,h in enumerate(Hg):
        if len(h.ops) >= n:
            big_groups.append(i)
    return big_groups

p = 1
def ansatz(params, qubits,  depth=p):
    for q in range(qubits):
        qml.U3(params[3*q],params[3*q+1],params[3*q+2], wires=q)
    for d in range(1,depth+1):
        for q in range(qubits-1):
            qml.CNOT(wires=[q,q+1])
        for q in range(qubits):
            qml.U3(params[3*(d*qubits+q)], params[3*(d*qubits+q)+1], params[3*(d*qubits+q)+2], wires=q)

#def cost(x):
#    E = 0
#    for i,h in enumerate(Hg):
#        if i in big_groups:
#            E += evaluate_eigenstate_MM(sample_circuit(x, Measurement_list[i]),\
#                                        h, memory=M[i], memory_states=5000)
#        else:
#            E += evaluate_eigenstate(sample_circuit(x, Measurement_list[i]), h)
#    return E


def cost(x):
    E = 0
    for i,h in enumerate(Hg):
        E += evaluate_eigenstate_MM(sample_circuit(x, Measurement_list[i]),\
                                        h, memory=M[i], memory_states=5000)
    return E

def cost0(x):
    E = 0
    for i,h in enumerate(Hg):
        E += evaluate_eigenstate(sample_circuit(x, Measurement_list[i]), h)
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

    moleculars = ["H2", "H4", "LiH", "H2O"]
    qubits = [4, 8, 12, 14]
    max_itr = 100
    gradient_method = 'parameter_shift'
    # Read initial parameters
    init_params = read_initial_parameters()[num_exp-1]
  
    for i, mol in enumerate(moleculars):
        Hg = read_Hamiltonian(mol,'QWC')
        N = qubits[i]
        # Get groups with # of ops >= N
        big_groups = check_big_groups(Hg, N)
        # Get measurements for each group
        Measurement_list = get_measurement_list(Hg, N)

        dev = qml.device("lightning.qubit", wires=N, shots=100*N)

        @qml.qnode(dev)
        def sample_circuit(params, obs):
            ansatz(params, N)
            measurement_rotation(obs)
            return qml.counts()


        params = init_params[:3*(p+1)*N]
        if is_MM :
            start = time.process_time()
            M = [{} for _ in range(len(Hg))]
            for itr in range(max_itr):
                params = GradientDescent(cost, params, gradient_method, learning_rate=0.05)
                obj_value = cost(params)
            end = time.process_time()
        else:
            start = time.process_time()
            for itr in range(max_itr):
                params = GradientDescent(cost0, params, gradient_method, learning_rate=0.05)
                obj_value = cost0(params)   
            end = time.process_time()    
        time_used = np.round(end-start, 3)

        if is_MM:
            path = "Molecular_MM{}.txt".format(num_exp)
        else:
            path = "Molecular_normal{}.txt".format(num_exp)
        overwrite = False
        if i == 0 : overwrite = True
        write_results(path, mol, N, time_used, overwrite)
