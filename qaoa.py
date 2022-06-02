from cmath import isclose
from qiskit.quantum_info.operators import Operator, Pauli
from qiskit.algorithms import QAOA
from qiskit.opflow import SummedOp,PrimitiveOp,I
import numpy as np

# returns the bit index for an alpha and j
def bit(ncities,alpha, j):
    return j * ncities + alpha

def D(ncities,alpha, j):
    qubit = bit(ncities,alpha, j)
    nqubits = ncities**2

    pauli_string = ''.join(['I' if i!=qubit else 'Z' for i in range(nqubits-1,-1,-1) ])

    return 0.5 * PrimitiveOp(Pauli(''.join(['I' for _ in range(nqubits-1,-1,-1) ]))) - PrimitiveOp(Pauli(pauli_string))

"""
weights and connections are numpy matrixes that define a weighted and unweighted graph
penalty: how much to penalize longer routes (penalty=0 pays no attention to weight matrix)
"""
def build_cost(penalty, ncities, G):
    nqubits = ncities**2
    identity_string = ''.join(['I' for _ in range(nqubits-1,-1,-1) ])

    ret = PrimitiveOp(Pauli(identity_string))
    # constraint (a)
    for i in range(ncities):
        cur = PrimitiveOp(Pauli(identity_string))
        for j in range(ncities):
            cur -= D(ncities,i, j)
        ret += cur.compose(cur) 

    # constraint (b)
    for i in range(ncities):
        cur = PrimitiveOp(Pauli(identity_string))
        for j in range(ncities):
            cur -= D(ncities,j, i)
        ret += cur.compose(cur) 

    # constraint (d) (the weighting)
    for i in range(ncities-1):
        cur = PrimitiveOp(Pauli(identity_string))
        for j in range(ncities):
            for k in range(ncities):
                if not np.isclose(G[j, k],0.0):
                    cur -= D(ncities,j, i).compose(D(ncities,k, i+1),front=True) * G[j, k]
        ret += cur * penalty
        
    return ret.reduce()


