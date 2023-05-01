import sys
import time
import pennylane as qml
from pennylane import numpy as np
from measurement_memory import evaluate_eigenstate, evaluate_eigenstate_MM, GradientDescent


def random_Ising_H(N):
    coeffs = np.random.rand((N*(N-1)//2))
    obs = [qml.PauliZ(i)@qml.PauliZ(j) for i in range(N) for j in range(i+1, N)]
    H = qml.Hamiltonian(coeffs, obs, grouping_type='qwc')
    return H

p = 1
def ansatz(params, qubits, depth=p):
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
        argList.remove("-MM")
    # Number of experiment
    num_exp = int(argList[0])
    # The maximum problem size
    N_max = 20

    problem_sizes = range(2, N_max+1, 2)
    max_itr = 200
    gradient_method = 'parameter_shift'
    # Read initial parameters
    init_params = read_initial_parameters()[num_exp-1]

    time_used = []
    for N in problem_sizes:
        H = random_Ising_H(N)
        dev = qml.device("lightning.qubit", wires=N, shots=1000*N**2)

        @qml.qnode(dev)
        def sample_circuit(params):
            ansatz(params, N)
            return qml.counts()

        params = init_params[:(p+1)*N]
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
        path = "Ising_MM{}.txt".format(num_exp)
    else:
        path = "Ising_normal{}.txt".format(num_exp)
    write_results(path, problem_sizes, time_used)
