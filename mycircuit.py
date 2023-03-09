import pennylane as qml

## RY (CNOT RY)*depth
def hardware_efficient_RY(params, qubits, depth=1):
    if not len(params) == (depth+1)*qubits:
        raise Exception("Length of params vector didn't match the circuit parameters number.")

    for q in range(qubits):
            qml.RY(params[q], wires=q)
    for d in range(1,depth+1):
        for q in range(qubits-1):
            qml.CNOT(wires=[q,q+1])
        for q in range(qubits):
            qml.RY(params[d*qubits+q], wires=q)
