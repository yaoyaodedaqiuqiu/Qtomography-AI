import numpy as np
import pickle
from scipy.linalg import expm

# Pauli Matrices
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
I = np.eye(2, dtype=complex)

# Function to build the Ising Hamiltonian
def build_ising_hamiltonian(J, h, n_qubits):
    """
    Constructs the Ising Hamiltonian for a given number of qubits.
    J: Coupling strength array (length n_qubits - 1)
    h: Transverse field strength array (length n_qubits)
    n_qubits: Number of qubits
    """
    H = np.zeros((2**n_qubits, 2**n_qubits), dtype=complex)
    for i in range(n_qubits - 1):
        H -= J[i] * kron_n([I]*i + [Z, Z] + [I]*(n_qubits - i - 2))
    for i in range(n_qubits):
        H -= h[i] * kron_n([I]*i + [X] + [I]*(n_qubits - i - 1))
    return H

# Kronecker product for n matrices
def kron_n(matrices):
    result = matrices[0]
    for mat in matrices[1:]:
        result = np.kron(result, mat)
    return result

# Function to simulate a random Pauli measurement
def measure_random_pauli(qubits_state, n_measure, n_qubits):
    """
    Simulates a single measurement on randomly chosen qubits with random Pauli operators.
    qubits_state: State vector of the system
    n_measure: List of qubits to measure
    n_qubits: Total number of qubits
    """
    pauli_ops = [X, Y, Z]
    measurement_results = []
    for qubit in n_measure:
        # Randomly choose a Pauli operator
        pauli = pauli_ops[np.random.randint(0, 3)]
        # Construct the full measurement operator
        measurement_op = kron_n([I]*qubit + [pauli] + [I]*(n_qubits - qubit - 1))
        # Compute the expectation value
        expectation_value = np.vdot(qubits_state, measurement_op @ qubits_state).real
        # Simulate a binary measurement outcome (+1 or -1) based on the expectation value
        outcome = np.random.choice([1, -1], p=[(1 + expectation_value)/2, (1 - expectation_value)/2])
        measurement_results.append(outcome)
    return measurement_results

def generate_ising_dataset(n_qubits, n_samples):
    """
    Generates a dataset for the 9-qubit Ising model.
    n_qubits: Number of qubits
    n_samples: Number of samples to generate
    """
    dataset = []
    for _ in range(n_samples):
        J = np.random.uniform(-1, 1, n_qubits - 1)  # Random coupling strengths
        h = np.random.uniform(-1, 1, n_qubits)  # Random transverse fields
        H = build_ising_hamiltonian(J, h, n_qubits)
        eigenvalues, eigenvectors = np.linalg.eigh(H)
        ground_state = eigenvectors[:, np.argmin(eigenvalues)]  # Ground state
        # Extract the minimum eigenvalue (for simplicity, using this as the target)
        min_eigenvalue = eigenvalues[0]

        # Randomly choose 3 qubits to measure
        n_measure = np.random.choice(range(n_qubits), size=3, replace=False)
        measurement_result = measure_random_pauli(ground_state, n_measure, n_qubits)
        
        # Append (measurement qubits, measurement result, Hamiltonian, and target eigenvalue)
        dataset.append((n_measure, measurement_result, H, min_eigenvalue))
        
    # Save dataset to a file
    with open('Ising_dataset.pkl', 'wb') as f:
        pickle.dump(dataset, f)

n_qubits = 9
n_samples = 100
generate_ising_dataset(n_qubits, n_samples)
