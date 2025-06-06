# dataset.py
import os
import numpy as np
import torch 
import pickle
from scipy.linalg import expm
import matplotlib.pyplot as plt
import logging

# 初始化 logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')
logger = logging.getLogger(__name__)

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

class HeisenbergHamiltonian(Hamiltonian):
    """
    Heisenberg Hamiltonian class inheriting from the base Hamiltonian.
    """
    def __init__(self, n_qubits, J=None):
        """
        Initialize the Heisenberg Hamiltonian with optional coupling strengths (J).
        
        Parameters:
            n_qubits (int): Number of qubits.
            J (float or np.ndarray, optional): Coupling strength. If None, a random value is used.
        """
        super().__init__(n_qubits)
        self.J = J if J is not None else np.random.uniform(-1, 1, n_qubits - 1)
    
    def build_hamiltonian(self):
        """
        Constructs the Heisenberg Hamiltonian for the system.
        
        Returns:
            np.ndarray: The Hamiltonian matrix.
        """
        H = np.zeros((2**self.n_qubits, 2**self.n_qubits), dtype=complex)
        for i in range(self.n_qubits - 1):
            # Create the Heisenberg interaction terms (σ_i^x σ_(i+1)^x, σ_i^y σ_(i+1)^y, σ_i^z σ_(i+1)^z)
            for sigma_pair in [X, Y, Z]:
                H -= self.J[i] * kron_n([I]*i + [sigma_pair, sigma_pair] + [I]*(self.n_qubits - i - 2))
        self.H = H
        return self.H

class ClusterHamiltonian(Hamiltonian):
    """
    Cluster Hamiltonian class inheriting from the base Hamiltonian.
    """
    def __init__(self, n_qubits, J=None, h=None):
        """
        Initialize the Cluster Hamiltonian with optional coupling strengths (J) and transverse field strengths (h).
        
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
        Constructs the Cluster Hamiltonian for the system.
        
        Returns:
            np.ndarray: The Hamiltonian matrix.
        """
        H = np.zeros((2**self.n_qubits, 2**self.n_qubits), dtype=complex)
        for i in range(self.n_qubits - 1):
            # Create the Cluster interaction terms (σ_i^x σ_(i+1)^x, σ_i^y σ_(i+1)^y)
            for sigma_pair in [X, Y]:
                H -= self.J[i] * kron_n([I]*i + [sigma_pair, sigma_pair] + [I]*(self.n_qubits - i - 2))
        for i in range(self.n_qubits):
            # Add the transverse field terms (σ_i^z)
            H -= self.h[i] * kron_n([I]*i + [Z] + [I]*(self.n_qubits - i - 1))
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
        Z_trace = np.trace(exp_minus_beta_H)
        rho = exp_minus_beta_H / Z_trace
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

def generate_and_save_data(N, J_list, h_list, t_list, model_type='Ising', data_folder='correlation_dataset'):
    """
    Generate Gibbs states for each combination of h, j, t and save correlation data.
    
    Parameters:
        N (int): Number of qubits.
        J_list (list of float): List of j values.
        h_list (list of float): List of h values.
        t_list (list of float): List of temperature values.
        model_type (str): The model to generate data for ('Ising', 'Heisenberg', or 'Cluster').
        data_folder (str): Folder to save the generated data.
    """
    # Define subfolders based on the data_folder and model_type
    model_folder = os.path.join(data_folder, model_type.lower())
    gibbs_folder = os.path.join(model_folder, 'gibbs_dataset')
    correlation_folder = os.path.join(model_folder, 'correlation_dataset')
    
    # Ensure target folders exist
    os.makedirs(gibbs_folder, exist_ok=True)
    os.makedirs(correlation_folder, exist_ok=True)
    
    correlation_data = {}
    
    # Iterate over all combinations of h, j, t
    for t in t_list:
        for j in J_list:
            for h in h_list:
                # Construct h array: all h_i are equal to current h
                h_array = np.full(N, h, dtype=np.float32)
                J_array = np.full(N-1, j, dtype=np.float32)
                
                # Initialize Hamiltonian and simulator based on the selected model type
                if model_type == 'Ising':
                    hamiltonian = IsingHamiltonian(n_qubits=N, J=J_array, h=h_array)
                elif model_type == 'Heisenberg':
                    hamiltonian = HeisenbergHamiltonian(n_qubits=N, J=J_array)
                elif model_type == 'Cluster':
                    hamiltonian = ClusterHamiltonian(n_qubits=N, J=J_array, h=h_array)
                else:
                    raise ValueError(f"Unsupported model type: {model_type}")
                
                simulator = GibbsStateSimulator(hamiltonian)
                
                # Simulate Gibbs state at temperature T
                gibbs_states = simulator.simulate_temperature_scan(t, t + 1, 1)  # Only T=t
                
                # Save Gibbs states to pickle
                gibbs_filename = f'gibbs_{model_type.lower()}_nq{N}_T{t}_j{j:.4f}_h{h:.4f}.pkl'
                gibbs_filepath = os.path.join(gibbs_folder, gibbs_filename)
                simulator.save_gibbs_states_to_pkl(gibbs_states, gibbs_filepath)
                print(f"Saved Gibbs states to: {gibbs_filepath}")
                
                # Calculate correlation term for T=t
                gibbs_state = gibbs_states[t]
                corr_term_each = calculate_correlation(gibbs_state, N)
                
                # Save correlation terms to numpy file
                correlation_filename = f'gibbs_{model_type.lower()}_nq{N}_T{t}_j{j:.4f}_h{h:.4f}_Z.npy'
                correlation_filepath = os.path.join(correlation_folder, correlation_filename)
                np.save(correlation_filepath, corr_term_each)
                print(f"Saved correlation terms to: {correlation_filepath}")
                
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

def is_number(s):
    """
    Helper function to check if a string can be converted to float.
    """
    try:
        float(s)
        return True
    except ValueError:
        return False

class CorrelationDatasetPyTorch:
    def __init__(self, data_folder, N, model_type='Ising'):
        """
        Initialize the dataset.
        
        Parameters:
            data_folder (str): Path to the correlation data folder.
            N (int): Number of qubits.
            model_type (str): The model to load data for ('Ising', 'Heisenberg', or 'Cluster').
        """
        self.inputs = []   # Store input features [h, j, t]
        self.targets = []  # Store correlation terms
        
        # Define the folder for the chosen model type
        model_folder = os.path.join(data_folder, model_type.lower())
        correlation_folder = os.path.join(model_folder, 'correlation_dataset')
        
        # Traverse all files in the correlation folder
        for filename in os.listdir(correlation_folder):
            if filename.startswith(f'gibbs_{model_type.lower()}_nq{N}_T') and filename.endswith('_Z.npy'):
                # Extract h, j, t values from filename
                try:
                    parts = filename.replace('.npy', '').split('_')
                    t = None
                    j = None
                    h = None
                    for part in parts:
                        if part.startswith('T') and is_number(part[1:]):
                            t = float(part[1:])
                        elif part.startswith('j') and is_number(part[1:]):
                            j = float(part[1:])
                        elif part.startswith('h') and is_number(part[1:]):
                            h = float(part[1:])
                    if t is None or j is None or h is None:
                        raise ValueError("Missing h, j, or t values.")
                except (ValueError, IndexError) as e:
                    logger.warning(f"Unable to parse h, j, t values from filename: {filename}")
                    continue
                
                # Load correlation data
                filepath = os.path.join(correlation_folder, filename)
                corr = np.load(filepath)
                
                # Add h, j, t and correlation terms to lists
                self.inputs.append([h, j, t])
                self.targets.append(corr)
        
        self.inputs = np.array(self.inputs, dtype=np.float32)    # Shape: (number of samples, 3)
        self.targets = np.array(self.targets, dtype=np.float32)  # Shape: (number of samples, N-1)
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        input_features = self.inputs[idx]
        target = self.targets[idx]
        return torch.tensor(input_features, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)

if __name__ == "__main__":
    # Example usage of the data generation module
    N = 9
    J_list = [1.0, 1.5, 2.0]  # Example j values
    h_list = np.linspace(0, 1, 21)  # 21 h values from 0 to 1
    t_list = [1.0, 1.5, 2.0]  # Example temperature values
    
    generate_and_save_data(N, J_list, h_list, t_list, data_folder='correlation_dataset')
