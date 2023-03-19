import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

## Evaluate H
def evaluate_eigenstate_MM(sample, G, memory, memory_states=10000):
    total = 0
    G_val = 0
    coeffs = G.coeffs
    for k,v in sample.items():
        try:
            G_val += memory[k]*v
            
        except:
            eigvals = [eigenvalue(c) for c in k]
            
            kGk = 0
            for i,op in enumerate(G.ops):
                if op.name == 'Identity':
                    kGk += coeffs[i]
                else: 
                    ws = op.wires.tolist()
                    op_val = 1
                    for w in ws: 
                        op_val *= eigvals[w]
                    kGk += coeffs[i]*op_val
            memory[k] = kGk
            G_val += kGk*v
                    
        total += v
    return G_val/total

def evaluate_eigenstate(sample, G):
    total = 0
    G_val = 0
    coeffs = G.coeffs
    for k,v in sample.items():
        eigvals = [eigenvalue(c) for c in k]
        kGk = 0
        for i,op in enumerate(G.ops):
            if op.name == 'Identity':
                kGk += coeffs[i]
            else:
                ws = op.wires.tolist()
                op_val = 1
                for w in ws:
                    op_val *= eigvals[w]
                kGk += coeffs[i]*op_val
        G_val += kGk*v
        total += v
    return G_val/total

def eigenvalue(state:str):
    return int(1-2*int(state))


## Hamiltonian grouping 
def group_H(H):
    Hg = []
    for g in H.grouping_indices:
        obs = [H.ops[i] for i in g]
        coeffs = [H.coeffs[i] for i in g]
        Hg.append(qml.Hamiltonian(coeffs, obs, grouping_type='qwc'))
    return Hg

def get_measurement_list(Hg:list, qubits):
    Measurement_list = []
    for g in Hg:
        basis = [0]*qubits
        for op in g.ops:
            wires = op.wires.labels
            Paulis = op.name
            if not Paulis == 'Identity':
                if type(Paulis) == str:
                    Paulis = [Paulis]
                
                for i,obs in enumerate(Paulis):
                    if basis[wires[i]] == 0 :
                        basis[wires[i]] = obs
                    else:
                        assert(basis[wires[i]] == obs), "Not qubit-wise commuted!"
        Measurement_list.append(basis)
    return Measurement_list

def measurement_rotation(M):
    for i,m in enumerate(M):
        if m == 'PauliX':
            qml.Hadamard(i)
        elif m == 'PauliY':
            qml.adjoint(qml.S(i))
            qml.Hadamard(i)


# Converter
def char_to_Pauli(c:str, i:int):
    if c == 'X':
        return qml.PauliX(i)
    elif c == 'Y':
        return qml.PauliY(i)
    elif c == 'Z':
        return qml.PauliZ(i)
    else:
        return None

def str_to_Pauli(H:list):
    ops = []
    for string in H:
        obs = None
        for i,c in enumerate(string):
            if not obs:
                obs = char_to_Pauli(c,i)
            else:
                if char_to_Pauli(c,i):
                    obs = obs @ char_to_Pauli(c,i)
        ops.append(obs)
    return ops
        
            

## Calculate Gradient
def finite_difference(cost, x, delta=0.0001):
    grad = []
    for i in range(len(x)):
        x1 = [j for j in x]
        x2 = [j for j in x]
        x1[i] = x1[i] + delta
        x2[i] = x2[i] - delta
        grad.append( (cost(x1)-cost(x2))/(2*delta) )
    return np.array(grad)

def parameter_shift(cost, x):
    grad = []
    for i in range(len(x)):
        x1 = [j for j in x]
        x2 = [j for j in x]
        x1[i] = x1[i] + 0.5*np.pi
        x2[i] = x2[i] - 0.5*np.pi
        grad.append(0.5*(cost(x1)-cost(x2)))
    return np.array(grad)


## Classical optmizer
def GradientDescent(cost, params, gradient=None, learning_rate=0.01):
    if gradient == "finite_difference":
        G = finite_difference(cost, params)
    elif gradient == "parameter_shift":
        G = parameter_shift(cost, params)
    else:
        raise Exception("Not supported gradient type.")
    
    new_params = [ params[i]-learning_rate*G[i] for i in range(len(params)) ]
    return new_params






