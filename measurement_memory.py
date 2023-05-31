import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

## Evaluate H
def evaluate_eigenstate_MM(sample, G, memory):
    # Input: circuit sample result, 
    #        G: A group of commuting operators in tuple ([Pauli operators], [coefficients]), 
    #        measurement memory
    # Output : Expactation value <h> evaluated w.r.t. the sample
    total = 0
    G_val = 0
    ops = G[0]
    coeffs = G[1]
    for k,v in sample.items():
        try:
            G_val += memory[k]*v
        except:
            eigvals = [eigenvalue(c) for c in k]            
            kGk = 0
            for i,op in enumerate(ops):
                op_val = 1
                for j, pw in enumerate(op):
                    if not pw == 'I':
                        op_val *= eigvals[j]
                kGk += coeffs[i]*op_val 
            memory[k] = kGk
            G_val += kGk*v
                    
        total += v
    return G_val/total

def evaluate_eigenstate(sample, G):
    # Input: circuit sample result,
    #        G: A group of commuting operators in tuple ([Pauli operators], [coefficients]),
    # Output : Expactation value <G> evaluated w.r.t. the sample
    total = 0
    G_val = 0
    ops = G[0]
    coeffs = G[1]
    for k,v in sample.items():
        eigvals = [eigenvalue(c) for c in k]
        kGk = 0
        for i,op in enumerate(ops):
            op_val = 1
            for j, pw in enumerate(op):
                if not pw == 'I':
                    op_val *= eigvals[j]
            kGk += coeffs[i]*op_val
        G_val += kGk*v
        total += v
    return G_val/total

def eigenvalue(state:str):
    return int(1-2*int(state))



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
        if not obs: obs = qml.Identity(0)
        ops.append(obs)
    return ops


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
            

## Gradient Method
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


