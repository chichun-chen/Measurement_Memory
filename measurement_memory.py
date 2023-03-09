import numpy as np

## Evaluate H
def evaluate_eigenstate_MM(sample, H, memory, memory_states=10000):
    total = 0
    H_val = 0
    coeffs = H.coeffs
    for k,v in sample.items():
        try:
            H_val += memory[k]*v
            
        except:
            eigvals = [eigenvalue(c) for c in k]
            
            kHk = 0
            for i,op in enumerate(H.ops):
                ws = op.wires.tolist()
                op_val = 1
                for w in ws: 
                    op_val *= eigvals[w]
                kHk += coeffs[i]*op_val
            H_val += kHk*v
            memory[k] = kHk
                    
        total += v
    return H_val/total

def eigenvalue(state:str):
    return int(1-2*int(state))


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






