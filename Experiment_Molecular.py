import os
import sys
import time
import pennylane as qml
from pennylane import numpy as np
from measurement_memory import evaluate_eigenstate, evaluate_eigenstate_MM,\
                               GradientDescent, str_to_Pauli
from measurement_memory import get_measurement_list, measurement_rotation


def read_Hamiltonian(mol:str):
    path = os.path.abspath(os.path.join("Hamiltonians", "{}_Hamiltonian.txt".format(mol)))
    Hg = []
    with open(path) as f:
        group_ops = []
        group_coeffs = []
        for line in f.readlines():
            s = line.split(' ')
            if s[0] == '\n' or s[0] == None:
                Hg.append(qml.Hamiltonian(group_coeffs, str_to_Pauli(group_ops), grouping_type='qwc'))
                group_ops = []
                group_coeffs = []
            else:
                group_ops.append(s[0])
                if s[1][-2:] == '\n':
                    group_coeffs.append(float(s[1][:-2]))
                else:
                    group_coeffs.append(float(s[1])) 
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

def cost(x):
    E = 0
    for i,h in enumerate(Hg):
        if i in big_groups:
            E += evaluate_eigenstate_MM(sample_circuit(x, Measurement_list[i]),\
                                        h, memory=M[i], memory_states=5000)
        else:
            E += evaluate_eigenstate(sample_circuit(x, Measurement_list[i]), h)
    return E

def cost0(x):
    E = 0
    for i,h in enumerate(Hg):
        E += evaluate_eigenstate(sample_circuit(x, Measurement_list[i]), h)
    return E


def write_results(path, mols, Ns, times):
    f = open(path, 'w')
    f.write("Mol N Time\n")
    for i in range(len(Ns)):
        f.write("{} {} {}\n".format(mols[i], Ns[i], times[i]))
    f.close()

if __name__ == '__main__':
    argList = sys.argv[1:]
    # if used MM
    is_MM = False
    if  "-MM" in argList:
        is_MM = True
        argList.remove("-MM")
    # Number of experiment
    num_exp = argList[0]    

    moleculars = ["H2", "LiH", "BeH2", "H2O", "O2"]
    qubits = [2, 4, 6, 8, 12]
    max_itr = 300
    gradient_method = 'parameter_shift'

    time_used = []
    for i, mol in enumerate(moleculars):
        Hg = read_Hamiltonian(mol)
        N = qubits[i]
        # Get groups with # of ops >= N
        big_groups = check_big_groups(Hg, N)
        # Get measurements for each group
        Measurement_list = get_measurement_list(Hg, N)

        dev = qml.device("default.qubit", wires=N, shots=10000)

        @qml.qnode(dev)
        def sample_circuit(params, obs):
            ansatz(params, N)
            measurement_rotation(obs)
            return qml.counts()

        init_params = np.random.rand(3*(p+1)*N)
        params = init_params
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
        time_used.append(np.round(end-start, 3))

    if is_MM:
        path = "Molecular_MM{}.txt".format(num_exp)
    else:
        path = "Molecular_normal{}.txt".format(num_exp)
    write_results(path, moleculars, qubits, time_used)
