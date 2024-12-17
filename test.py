# test.py
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import logging
from dataset import generate_and_save_data  
from model import CorrelationModel  

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')
logger = logging.getLogger()

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Define PyTorch Dataset class for loading h, j, t and corresponding correlation terms
class CorrelationDatasetPyTorch(Dataset):
    def __init__(self, data_folder, N):
        """
        Initialize the dataset.
        
        Parameters:
            data_folder (str): Path to the correlation data folder.
            N (int): Number of qubits.
        """
        self.inputs = []   # Store input features [h, j, t]
        self.targets = []  # Store correlation terms
        
        correlation_folder = os.path.join(data_folder, 'correlation_dataset')
        
        # Traverse all files in the correlation folder
        for filename in os.listdir(correlation_folder):
            if filename.startswith(f'gibbs_ising_nq{N}_T') and filename.endswith('_Z.npy'):
                # Extract h, j, t values from filename
                try:
                    parts = filename.replace('.npy', '').split('_')
                    t = None
                    j = None
                    h = None
                    for part in parts:
                        if part.startswith('T'):
                            t = float(part[1:])
                        elif part.startswith('j'):
                            j = float(part[1:])
                        elif part.startswith('h'):
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

def main():
    # Parameter settings for test data
    N = 9
    J_list = [1.0, 1.5, 2.0]          # Same j values as training
    h_list = np.linspace(0, 1, 6)     # 6 h values from 0 to 1 with a step size of 0.2
    t_list = [1.2, 1.8, 2.5]          # New temperature values for testing
    
    # Data generation
    logger.info("Generating and saving test data...")
    generate_and_save_data(N, J_list, h_list, t_list, data_folder='test_dataset')
    logger.info("Test data generation completed.")
    
    # Create test dataset
    data_folder = 'test_dataset'
    test_dataset = CorrelationDatasetPyTorch(data_folder, N)
    if len(test_dataset) == 0:
        logger.error("No test data found in the dataset. Please check the folder path and file format.")
        return

    logger.info(f"Test Dataset size: {len(test_dataset)}")
    
    # Create data loader
    batch_size = 32
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    output_dim = N - 1  # Number of correlations
    model = CorrelationModel(input_dim=3, hidden_dim=128, output_dim=output_dim).to(device)
    
    # Load the trained model parameters
    checkpoint_path = 'model_pth/best_correlation_model.pth'
    if not os.path.exists(checkpoint_path):
        logger.error(f"Model checkpoint '{checkpoint_path}' not found.")
        return
    
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    logger.info(f"Loaded model from '{checkpoint_path}'")
    
    # Define loss function
    criterion = nn.MSELoss()
    
    # Evaluate the model
    with torch.no_grad():
        total_loss = 0.0
        all_inputs = []
        all_targets = []
        all_predictions = []
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            
            all_inputs.append(inputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_predictions.append(outputs.cpu().numpy())
        
        average_loss = total_loss / len(test_dataset)
        logger.info(f"Test MSE Loss: {average_loss:.6f}")
    
    # Concatenate all batches
    all_inputs = np.vstack(all_inputs)
    all_targets = np.vstack(all_targets)
    all_predictions = np.vstack(all_predictions)
    
    # Extract h, j, t
    h_vals = all_inputs[:, 0]
    j_vals = all_inputs[:, 1]
    t_vals = all_inputs[:, 2]
    
    # Get unique j and t values
    unique_j = np.unique(j_vals)
    unique_t = np.unique(t_vals)
    correlation_names = [f'Correlation {i+2}' for i in range(output_dim)]
    

    figure_test_folder = 'figure'
    
    # Create a plot for each t value
    for t in unique_t:
        plt.figure(figsize=(20, 15))
        plt.suptitle(f'Predicted vs True Correlations at Temperature T={t}', fontsize=16)
        
        for i in range(output_dim):
            plt.subplot(4, 2, i+1)  # 4 rows x 2 columns layout, adjust based on output_dim if necessary
            for j in unique_j:
                # Select data for the current t and j
                mask = (t_vals == t) & (j_vals == j)
                if not np.any(mask):
                    continue
                h_subset = h_vals[mask]
                target_subset = all_targets[mask, i]
                pred_subset = all_predictions[mask, i]
                plt.plot(h_subset, target_subset, label=f'True j={j}', linestyle='-', marker='o')
                plt.plot(h_subset, pred_subset, label=f'Predicted j={j}', linestyle='--', marker='x')
            
            plt.xlabel('h')
            plt.ylabel(r'$\langle \sigma_1^z \sigma_j^z \rangle$')
            plt.title(correlation_names[i])
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
        figure_path = os.path.join(figure_test_folder, f'correlations_T{t:.1f}.png')
        plt.savefig(figure_path)
        logger.info(f"Saved plot: {figure_path}")
        plt.close()
    
    logger.info("Testing completed.")

if __name__ == "__main__":
    main()
