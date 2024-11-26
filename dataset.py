import numpy as np
import pickle
from scipy.linalg import expm


X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
I = np.eye(2, dtype=complex)

# Kronecker product for n matrices 
def kron_n(matrices):
    result = matrices[0]
    for mat in matrices[1:]:
        result = np.kron(result, mat)
    return result

# general Hamiltonian class
class Hamiltonian:
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits
        self.H = None  # Will store the Hamiltonian matrix
    
    def build_hamiltonian(self):
        raise NotImplementedError("This method should be implemented by subclasses")

# the Ising Hamiltonian class inheriting from Hamiltonian
class IsingHamiltonian(Hamiltonian):
    def __init__(self, n_qubits, J=None, h=None):
        """
        Initialize Ising Hamiltonian with optional coupling strengths (J) and transverse field strengths (h).
        
        n_qubits: Number of qubits.
        J: Coupling strengths (array of length n_qubits-1).
        h: Transverse field strengths (array of length n_qubits).
        """
        super().__init__(n_qubits)
        self.J = J if J is not None else np.random.uniform(-1, 1, n_qubits - 1)
        self.h = h if h is not None else np.random.uniform(-1, 1, n_qubits)
    
    def build_hamiltonian(self):
        """
        Constructs the Ising Hamiltonian for the system.
        
        This function sets the Hamiltonian matrix (self.H) using the current values of J and h.
        """
        H = np.zeros((2**self.n_qubits, 2**self.n_qubits), dtype=complex)
        for i in range(self.n_qubits - 1):
            H -= self.J[i] * kron_n([I]*i + [Z, Z] + [I]*(self.n_qubits - i - 2))
        for i in range(self.n_qubits):
            H -= self.h[i] * kron_n([I]*i + [X] + [I]*(self.n_qubits - i - 1))
        self.H = H
        return self.H

#  Gibbs State Simulator Class 
class GibbsStateSimulator:
    def __init__(self, hamiltonian: Hamiltonian):
        """
        Initializes the GibbsStateSimulator class with a given Hamiltonian.

        hamiltonian: An instance of a subclass of Hamiltonian (e.g., IsingHamiltonian).
        """
        self.hamiltonian = hamiltonian  # Can accept any Hamiltonian subclass instance

    def generate_gibbs_state(self, beta):
        """
        Generates the Gibbs state for the given Hamiltonian matrix at a given inverse temperature beta.
        
        beta: Inverse temperature (1 / k_B T).
        
        Returns: The Gibbs state matrix (rho).
        """
        exp_minus_beta_H = expm(-beta * self.hamiltonian.H)
        Z = np.trace(exp_minus_beta_H)
        rho = exp_minus_beta_H / Z
        return rho

    def simulate_gibbs_state(self, beta):
        """
        Simulates the Gibbs state for the system with the current Hamiltonian and given beta.
        
        beta: Inverse temperature (1 / k_B T).
        
        Returns: The Gibbs state matrix (rho).
        """
        self.hamiltonian.build_hamiltonian()
        rho = self.generate_gibbs_state(beta)
        
        return rho

    def simulate_temperature_scan(self, T_min, T_max, step):
        """
        Simulates the Gibbs states for a range of temperatures.

        T_min: Minimum temperature.
        T_max: Maximum temperature.
        step: Step size for temperature scan.
        
        Returns: A dictionary of temperatures to Gibbs states.
        """
        temperature_range = np.arange(T_min, T_max, step)
        gibbs_states = {}
        
        for T in temperature_range:
            beta = 1.0 / T  # Inverse temperature
            gibbs_state = self.simulate_gibbs_state(beta)
            gibbs_states[T] = gibbs_state
        
        return gibbs_states

    def save_gibbs_states_to_pkl(self, gibbs_states, filename):
        """
        Saves the Gibbs states to a pickle file.

        gibbs_states: Dictionary of temperature -> Gibbs state.
        filename: Name of the pickle file.
        """
        with open(filename, 'wb') as f:
            pickle.dump(gibbs_states, f)
        print(f"Gibbs states saved to {filename}")

# Example 
def example_simulation():
    n_qubits = 9  # Number of qubits
    T_min = 0.5  # Minimum temperature
    T_max = 3.0  # Maximum temperature
    step = 0.5  # Temperature step size

    # Create an IsingHamiltonian instance
    ising_hamiltonian = IsingHamiltonian(n_qubits)

    # Create a GibbsStateSimulator instance with the Ising Hamiltonian
    simulator = GibbsStateSimulator(ising_hamiltonian)

    # Simulate the Gibbs state over a temperature range
    gibbs_states = simulator.simulate_temperature_scan(T_min, T_max, step)
    
    for T, state in gibbs_states.items():
        print(f"Gibbs state at T = {T}:\n", state)

    # Save the Gibbs states to a pickle file
    simulator.save_gibbs_states_to_pkl(gibbs_states, 'gibbs_states.pkl')

# Run the example simulation
example_simulation()
