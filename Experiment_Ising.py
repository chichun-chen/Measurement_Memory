import sys
import time
import pennylane as qml
from pennylane import numpy as np
from measurement_memory import evaluate_eigenstate, evaluate_eigenstate_MM, GradientDescent
from Hamiltonians.utils import read_Hamiltonian

p = 1
def ansatz(params, qubits, depth=p):
    for q in range(qubits):
            qml.RY(params[q], wires=q)
    for d in range(1,depth+1):
        for q in range(qubits-1):
            qml.CNOT(wires=[q,q+1])
        for q in range(qubits):
            qml.RY(params[d*qubits+q], wires=q)

def cost_MM(x):
    # cost function for measurement memory
    return evaluate_eigenstate_MM(sample_circuit(x), Hg[0], memory=M)

def cost0(x):
    # cost function for normal evaluation
    return evaluate_eigenstate(sample_circuit(x), Hg[0])


def read_initial_parameters():
    try:
        initial_params = []
        with open("Ising_initial_parameters.txt", 'r') as f:
            for line in f.readlines():
                s = line.split(' ')
                if s[-1] == '\n' : s = s[:-1]
                for i,n in enumerate(s):
                    s[i] = float(n)
                initial_params.append(s)
    except:
        raise FileExistsError("Must generate initial parameters.")
    return initial_params

def get_file_name(is_MM:bool):
    if is_MM :
        return "cn11_Ising_MM{}.txt".format(num_exp)
    else:
        return "cn11_Ising_normal{}.txt".format(num_exp)

def write_results(path, N, time, overwrite=True):
    if overwrite:
        f = open(path, 'w') 
        f.write("N Time\n")
    else:
        f = open(path, 'a')
    f.write("{} {}\n".format(N, time))
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
    N_max = 14
    max_itr = 200
    gradient_method = 'parameter_shift'

    # Read initial parameters
    init_params = read_initial_parameters()[num_exp-1]
    problem_sizes = range(14, N_max+1, 2)

    for N in problem_sizes:
        Hg = read_Hamiltonian("Ising", N=N)
        dev = qml.device("lightning.qubit", wires=N, shots=200*N**2)

        @qml.qnode(dev)
        def sample_circuit(params):
            ansatz(params, N)
            return qml.counts()

        params = init_params[:(p+1)*N]
        if is_MM :
            start = time.process_time()
            M = {}
            for itr in range(max_itr):
                params = GradientDescent(cost_MM, params, gradient_method, learning_rate=0.05)
                obj_value = cost_MM(params)
            end = time.process_time()
        else:
            start = time.process_time()
            for itr in range(max_itr):
                params = GradientDescent(cost0, params, gradient_method, learning_rate=0.05)
                obj_value = cost0(params)
            end = time.process_time()    

        time_used = np.round(end-start, 3)

        path = get_file_name(is_MM)
        overwrite = False
        if N == problem_sizes[0] : overwrite = True
        write_results(path, N, time_used, overwrite)
