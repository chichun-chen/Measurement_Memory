import sys 
import numpy as np

def generate_initial_parameters(name, max_qubits, circuit_reps, rotation, sets):
    with open("{}_initial_parameters.txt".format(name), 'w') as f:
        for _ in range(sets):
            for _ in range((circuit_reps+1)*max_qubits*rotation):
                f.write("{} ".format(np.random.rand()))
            f.write("\n")


if __name__ == '__main__':
    argList = sys.argv[1:]
    if "-Mol" in argList: 
        name = "Mol"
        max_qubits = 14
        p = 1
        r = 3
        num_exp = 20
        generate_initial_parameters(name, max_qubits, p, r, num_exp)

    if "-Ising" in argList:
        name = "Ising"
        max_qubits = 20
        p = 1
        r = 1
        num_exp = 20
        generate_initial_parameters(name, max_qubits, p, r, num_exp)
    
