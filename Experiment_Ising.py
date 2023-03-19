import sys
import time
import pennylane as qml
from pennylane import numpy as np
from measurement_memory import evaluate_eigenstate, evaluate_eigenstate_MM, GradientDescent


def random_Ising_H(N):
    coeffs = np.random.rand((N*(N-1)//2 + N))
    obs = [qml.PauliZ(i)@qml.PauliZ(j) for i in range(N) for j in range(i, N)]
    H = qml.Hamiltonian(coeffs, obs, grouping_type='qwc')
    return H

def ansatz(params, depth=1):
    qubits = len(params)//2
    for q in range(qubits):
            qml.RY(params[q], wires=q)
    for d in range(1,depth+1):
        for q in range(qubits-1):
            qml.CNOT(wires=[q,q+1])
        for q in range(qubits):
            qml.RY(params[d*qubits+q], wires=q)

def cost(x):
    return evaluate_eigenstate_MM(sample_circuit(x), H, memory=M, memory_states=999999)

def cost0(x):
    return evaluate_eigenstate(sample_circuit(x), H)

def write_results(path, Ns, times):
    f = open(path, 'w')
    f.write("N Time\n")
    for i in range(len(Ns)):
        f.write("{} {}\n".format(Ns[i], times[i]))
    f.close()

if __name__ == '__main__':
    argList = sys.argv[1:]
    # if used MM
    is_MM = False
    if  "-MM" in argList:
        is_MM = True
    # The maximum problem size
    N_max = 4

    problem_sizes = range(2, N_max+1, 2)
    max_itr = 100
    gradient_method = 'parameter_shift'

    time_used = []
    for N in problem_sizes:
        H = random_Ising_H(N)
        dev = qml.device("default.qubit", wires=N, shots=10000)

        @qml.qnode(dev)
        def sample_circuit(params):
            ansatz(params)
            return qml.counts()

        init_params = np.random.rand(2*N)
        params = init_params
        if is_MM :
            start = time.process_time()
            M = {}
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
        path = "Ising_MM.txt"
    else:
        path = "Ising_normal.txt"
    write_results(path, problem_sizes, time_used)
