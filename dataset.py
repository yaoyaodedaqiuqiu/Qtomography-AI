import os
import numpy as np
import pickle
from scipy.linalg import expm
import matplotlib.pyplot as plt

# Define Pauli matrices
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
I = np.eye(2, dtype=complex)

def kron_n(matrices):
    """
    Compute the Kronecker product of a list of matrices.
    
    Parameters:
        matrices (list of np.ndarray): List of matrices to be Kronecker multiplied.
    
    Returns:
        np.ndarray: The resulting Kronecker product matrix.
    """
    result = matrices[0]
    for mat in matrices[1:]:
        result = np.kron(result, mat)
    return result

class Hamiltonian:
    """
    Base class for Hamiltonians.
    """
    def __init__(self, n_qubits):
        """
        Initialize the Hamiltonian.
        
        Parameters:
            n_qubits (int): Number of qubits in the system.
        """
        self.n_qubits = n_qubits
        self.H = None  # Hamiltonian matrix
    
    def build_hamiltonian(self):
        """
        Build the Hamiltonian matrix.
        This method should be implemented by subclasses.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

class IsingHamiltonian(Hamiltonian):
    """
    Ising Hamiltonian class inheriting from the base Hamiltonian.
    """
    def __init__(self, n_qubits, J=None, h=None):
        """
        Initialize the Ising Hamiltonian with optional coupling strengths (J) and transverse field strengths (h).
        
        Parameters:
            n_qubits (int): Number of qubits.
            J (np.ndarray, optional): Coupling strengths array of length n_qubits-1.
            h (np.ndarray, optional): Transverse field strengths array of length n_qubits.
        """
        super().__init__(n_qubits)
        self.J = J if J is not None else np.random.uniform(-1, 1, n_qubits - 1)
        self.h = h if h is not None else np.random.uniform(-1, 1, n_qubits)
    
    def build_hamiltonian(self):
        """
        Constructs the Ising Hamiltonian for the system.
        
        Returns:
            np.ndarray: The Hamiltonian matrix.
        """
        H = np.zeros((2**self.n_qubits, 2**self.n_qubits), dtype=complex)
        for i in range(self.n_qubits - 1):
            H -= self.J[i] * kron_n([I]*i + [Z, Z] + [I]*(self.n_qubits - i - 2))
        for i in range(self.n_qubits):
            H -= self.h[i] * kron_n([I]*i + [X] + [I]*(self.n_qubits - i - 1))
        self.H = H
        return self.H

class GibbsStateSimulator:
    """
    Gibbs State Simulator Class to generate Gibbs states for a given Hamiltonian.
    """
    def __init__(self, hamiltonian: Hamiltonian):
        """
        Initialize the GibbsStateSimulator with a given Hamiltonian.
        
        Parameters:
            hamiltonian (Hamiltonian): An instance of a Hamiltonian subclass.
        """
        self.hamiltonian = hamiltonian
    
    def generate_gibbs_state(self, beta):
        """
        Generates the Gibbs state for the given Hamiltonian matrix at a given inverse temperature beta.
        
        Parameters:
            beta (float): Inverse temperature (1 / k_B T).
        
        Returns:
            np.ndarray: The Gibbs state matrix (rho).
        """
        exp_minus_beta_H = expm(-beta * self.hamiltonian.H)
        Z = np.trace(exp_minus_beta_H)
        rho = exp_minus_beta_H / Z
        return rho
    
    def simulate_gibbs_state(self, beta):
        """
        Simulates the Gibbs state for the system with the current Hamiltonian and given beta.
        
        Parameters:
            beta (float): Inverse temperature (1 / k_B T).
        
        Returns:
            np.ndarray: The Gibbs state matrix (rho).
        """
        self.hamiltonian.build_hamiltonian()
        rho = self.generate_gibbs_state(beta)
        return rho
    
    def simulate_temperature_scan(self, T_min, T_max, step):
        """
        Simulates the Gibbs states for a range of temperatures.
        
        Parameters:
            T_min (float): Minimum temperature.
            T_max (float): Maximum temperature.
            step (float): Step size for temperature scan.
        
        Returns:
            dict: A dictionary mapping temperatures to Gibbs states.
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
        
        Parameters:
            gibbs_states (dict): Dictionary mapping temperatures to Gibbs states.
            filename (str): Name of the pickle file.
        """
        with open(filename, 'wb') as f:
            pickle.dump(gibbs_states, f)
        print(f"Gibbs states saved to {filename}")
    
    @staticmethod
    def load_gibbs_states_from_pkl(filename):
        """
        Load Gibbs states from a pickle file.
        
        Parameters:
            filename (str): Name of the pickle file.
        
        Returns:
            dict: Dictionary mapping temperatures to Gibbs states.
        """
        with open(filename, "rb") as f:
            gibbs_states = pickle.load(f)
        return gibbs_states

def calculate_correlation(gibbs_state, N):
    """
    Calculate the correlation for each pair of spins in the system.
    
    Parameters:
        gibbs_state (np.ndarray): The Gibbs state matrix (rho).
        N (int): Number of qubits.
    
    Returns:
        list: List of correlation terms <σ₁ᶻ σ_jᶻ> for j=2 to N.
    """
    corr_term_each = []
    for j in range(1, N):
        # Calculate correlation term for each pair (1, j)
        obs = [I] * N
        obs[0] = Z
        obs[j] = Z
        zz_obs = kron_n(obs)
        corr_term = np.trace(gibbs_state @ zz_obs).real  # Take the real part
        corr_term_each.append(corr_term)
    return corr_term_each

def generate_and_save_data(N, J_list, h_list, t_list):
    """
    Generate Gibbs states for each combination of h, j, t and save correlation data.
    
    Parameters:
        N (int): Number of qubits.
        J_list (list of float): List of j values.
        h_list (list of float): List of h values.
        t_list (list of float): List of temperature values.
    """
    correlation_folder = 'correlation_dataset'
    gibbs_folder = 'gibbs_dataset'
    model_pth_folder = 'model_pth'
    figure_folder = 'figure'
    # Ensure target folders exist
    if not os.path.exists(gibbs_folder):
        os.makedirs(gibbs_folder)
    if not os.path.exists(correlation_folder):
        os.makedirs(correlation_folder)
    if not os.path.exists(model_pth_folder):
        os.makedirs('model_pth', exist_ok=True)
    if not os.path.exists(figure_folder):
        os.makedirs('figure', exist_ok=True)
    
    correlation_data = {}
    
    # Iterate over all combinations of h, j, t
    for t in t_list:
        for j in J_list:
            for h in h_list:
                # Construct h array: all h_i are equal to current h
                h_array = np.full(N, h, dtype=np.float32)
                J_array = np.full(N-1, j, dtype=np.float32)
                
                # Initialize Hamiltonian and simulator
                ising_hamiltonian = IsingHamiltonian(n_qubits=N, J=J_array, h=h_array)
                simulator = GibbsStateSimulator(ising_hamiltonian)
                
                # Simulate Gibbs state at temperature T
                gibbs_states = simulator.simulate_temperature_scan(t, t + 1, 1)  # Only T=t
                
                # Save Gibbs states to pickle
                gibbs_filename = f'{gibbs_folder}/gibbs_ising_nq{N}_T{t}_j{j:.4f}_h{h:.4f}.pkl'
                simulator.save_gibbs_states_to_pkl(gibbs_states, gibbs_filename)
                print(f"Saved Gibbs states to: {gibbs_filename}")
                
                # Calculate correlation term for T=t
                gibbs_state = gibbs_states[t]
                corr_term_each = calculate_correlation(gibbs_state, N)
                
                # Save correlation terms to numpy file
                correlation_filename = f'{correlation_folder}/gibbs_ising_nq{N}_T{t}_j{j:.4f}_h{h:.4f}_Z.npy'
                np.save(correlation_filename, corr_term_each)
                print(f"Saved correlation terms to: {correlation_filename}")
                
                # Store in dictionary for plotting
                correlation_data[(h, j, t)] = corr_term_each
    
    # Optional: Plot correlations
    # Uncomment the following line if you want to generate plots during data generation
    # plot_correlation(h_list, J_list, t_list, N, correlation_data)

def plot_correlation(h_list, j_list, t_list, N, correlation_data):
    """
    Plot correlation for each combination of h, j, t.
    
    Parameters:
        h_list (list of float): List of h values.
        j_list (list of float): List of j values.
        t_list (list of float): List of temperature values.
        N (int): Number of qubits.
        correlation_data (dict): Dictionary with keys as (h, j, t) tuples and values as correlation lists.
    """
    plt.figure(figsize=(16, 12))
    subplot_idx = 1
    for t in t_list:
        for j in j_list:
            plt.subplot(len(t_list), len(j_list), subplot_idx)
            for h in h_list:
                key = (h, j, t)
                if key in correlation_data:
                    corr_term = correlation_data[key]
                    plt.plot(range(2, N + 1), corr_term, label=f'h={h:.2f}')
            plt.xlabel('j')
            plt.ylabel(r'$\langle \sigma_1^z \sigma_j^z \rangle$')
            plt.title(f'Temperature={t}, J={j}')
            plt.legend()
            subplot_idx += 1
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Example usage of the data generation module
    N = 9
    J_list = [1.0, 1.5, 2.0]  # Example j values
    h_list = np.linspace(0, 1, 21)  # 21 h values from 0 to 1 with step of 0.05
    t_list = [1.0, 1.5, 2.0]  # Example temperature values
    
    generate_and_save_data(N, J_list, h_list, t_list)
