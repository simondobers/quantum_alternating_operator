from qiskit import QuantumCircuit
from qiskit.quantum_info.operators import Operator, Pauli
from qiskit.opflow import SummedOp,PrimitiveOp
import numpy as np

def create_initial_state_circuit(num_nodes : int) ->QuantumCircuit:
    """Returns a curuit that creates an initial state for the TSP problem, where at t=i city i is visited

    Args:
        num_nodes (int): number of cities in the TSP problem

    Returns:
        QuantumCircuit: Quantum circuit that creates the inital state
    """
    nqubits = num_nodes**2
    circ = QuantumCircuit(nqubits)

    # visit i'th city at time i
    for i in range(num_nodes):
        circ.x(i*num_nodes + i)

    return circ

def S_Plus(num_nodes : int, city : int, time:int)-> Operator:
    """Creates S_Plus matrix for {city} at {time} according to equation (56) from https://arxiv.org/pdf/1709.03489.pdf

    Args:
        num_nodes (int):  number of cities in the TSP problem
        city (int): affected city
        time (int): point in time where city is visited

    Returns:
        Operator: The S_Plus Operator
    """
    nqubits = num_nodes**2
    qubit = city * num_nodes + time
    
    # define Operator where X acts on qubit {qubit}, else I
    pauli_x_string = ''.join(['I' if i!=qubit else 'X' for i in range(nqubits-1,-1,-1) ])

    # define Operator where i*Y acts on qubit {qubit}, else I
    pauli_y_string = ''.join(['I' if i!=qubit else 'Y' for i in range(nqubits-1,-1,-1) ])

    return PrimitiveOp(Pauli(pauli_x_string)) + 1j* PrimitiveOp(Pauli(pauli_y_string))


def S_Minus(num_nodes : int, city : int, time:int)-> Operator:
    """Creates S_Minus matrix for {city} at {time} according to equation (57) from https://arxiv.org/pdf/1709.03489.pdf

    Args:
        num_nodes (int):  number of cities in the TSP problem
        city (int): affected city
        time (int): point in time where city is visited

    Returns:
        Operator: The S_Plus Operator
    """
    nqubits = num_nodes**2
    qubit = city * num_nodes + time
    
    # define Operator where X acts on qubit {qubit}, else I
    pauli_x_string = ''.join(['I' if i!=qubit else 'X' for i in range(nqubits-1,-1,-1) ])

    # define Operator where i*Y acts on qubit {qubit}, else I
    pauli_y_string = ''.join(['I' if i!=qubit else 'Y' for i in range(nqubits-1,-1,-1) ])

    return PrimitiveOp(Pauli(pauli_x_string)) - 1j* PrimitiveOp(Pauli(pauli_y_string))

def create_mixer_operator(num_nodes:int)->Operator:
    """Create mixing Operator according to equation (54)-(58) from https://arxiv.org/pdf/1709.03489.pdf

    Args:
        num_nodes (int): number of cities in the TSP problem

    Returns:
        Operator: Mixer Operator
    """
    mixer_operators = []
    for t in range(num_nodes-1):
        for city_1 in range(num_nodes):
            for city_2 in range(num_nodes):
                # swap city_1 at t=i, city_2 at t=i+1 <-> city_1 at t=i+1, city_2 at t=i
                # if current state is city_1 at t=i, city_2 at t=i+1 (see eq. (58))
                i = t
                u = city_1
                v = city_2
                first_part = S_Plus(num_nodes, u, i)
                first_part = first_part.compose(S_Plus(num_nodes, v, i+1),front=True)
                first_part = first_part.compose(S_Minus(num_nodes, u, i+1),front=True)
                first_part = first_part.compose(S_Minus(num_nodes, v, i),front=True)

                second_part = S_Minus(num_nodes, u, i)
                second_part = second_part.compose(S_Minus(num_nodes, v, i+1),front=True)
                second_part = second_part.compose(S_Plus(num_nodes, u, i+1),front=True)
                second_part = second_part.compose(S_Plus(num_nodes, v, i),front=True)
                mixer_operators.append((first_part + second_part))
    

    return SummedOp(mixer_operators)

def create_phase_separator(graph:np.array) -> Operator:
    """Create Phase Separator according to equation (53) from https://arxiv.org/pdf/1709.03489.pdf 

    Args:
        graph (np.array): TSP graph, represented as symmetric np.array

    Returns:
        Operator: Phase Separation Operator
    """
    phase_separators = []
    ncities = graph.shape[0]
    nqubits = ncities**2

    for t in range(ncities - 1):
        for city_1 in range(ncities):
            for city_2 in range(ncities):

                # If these aren't the same city 
                distance = graph[city_1, city_2]
                if city_1 != city_2 and distance != 0.0:
                    qubit_1 = t * ncities + city_1
                    qubit_2 = (t + 1) * ncities + city_2

                    pauli_z1_string = ''.join(['I' if i!=qubit_1 else 'Z' for i in range(nqubits-1,-1,-1) ])
                    pauli_z2_string = ''.join(['I' if i!=qubit_2 else 'Z' for i in range(nqubits-1,-1,-1) ])

                    # Append with Z1 * Z2 (compose needs reverse order)
                    phase_separators.append(distance * PrimitiveOp(Pauli(pauli_z2_string)).compose(PrimitiveOp(Pauli(pauli_z1_string))))
    
    return SummedOp(phase_separators)