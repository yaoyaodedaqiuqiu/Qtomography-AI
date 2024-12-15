# train.py
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from dataset import generate_and_save_data  
from model import CorrelationModel  
import logging

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
        
        # Traverse all files in the data folder
        for filename in os.listdir(data_folder):
            if filename.startswith(f'gibbs_ising_nq{N}_T') and filename.endswith('_Z.npy'):
                # Extract h, j, t values from filename
                try:
                    parts = filename.split('_')
                    t = None
                    j = None
                    h = None
                    for part in parts:
                        if part.startswith('T'):
                            t = float(part[1:])
                        elif part.startswith('j'):
                            j = float(part[1:])
                        elif part.startswith('h'):
                            h_str = part[1:]
                            if h_str.endswith('.npy'):
                                h_str = h_str[:-4]
                            h = float(h_str)
                    if t is None or j is None or h is None:
                        raise ValueError("Missing h, j, or t values.")
                except (ValueError, IndexError) as e:
                    logger.warning(f"Unable to parse h, j, t values from filename: {filename}")
                    continue
                
                # Load correlation data
                filepath = os.path.join(data_folder, filename)
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
    # Parameter settings
    N = 9
    J_list = [1.0, 1.5, 2.0]          # Example j values
    h_list = np.linspace(0, 1, 21)    # 21 h values from 0 to 1 with a step size of 0.05
    t_list = [1.0, 1.5, 2.0]          # Example temperature values
    
    # Data generation (comment out after the first run)
    logger.info("Generating and saving data...")
    generate_and_save_data(N, J_list, h_list, t_list)
    logger.info("Data generation completed.")
    
    # Create dataset
    data_folder = 'correlation_dataset'
    dataset = CorrelationDatasetPyTorch(data_folder, N)
    if len(dataset) == 0:
        logger.error("No data found in the dataset. Please check the folder path and file format.")
        return

    logger.info(f"Dataset size: {len(dataset)}")
    
    # Split training and testing datasets (80% training, 20% testing)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    logger.info(f"Training set size: {len(train_dataset)}, Testing set size: {len(test_dataset)}")
    
    # Create data loaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model, loss function, and optimizer
    output_dim = N - 1  # Number of correlations
    model = CorrelationModel(input_dim=3, hidden_dim=128, output_dim=output_dim).to(device)
    criterion = nn.MSELoss()
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Model checkpoint path
    checkpoint_path = 'model_pth/best_correlation_model.pth'
    best_loss = float('inf')
    
    # Train the model
    num_epochs = 500
    logger.info("Starting training...")
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * inputs.size(0)
        
        epoch_loss /= train_size
        if (epoch + 1) % 50 == 0 or epoch == 0:
            logger.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.6f}")
        
        # Check if it's the best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"Best model updated at epoch {epoch+1} with loss {best_loss:.6f}")
    
    logger.info("Training completed.")
    
    # Evaluate the model
    model.eval()
    with torch.no_grad():
        total_loss = 0.0
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
        
        total_loss /= test_size
        logger.info(f"Test MSE Loss: {total_loss:.6f}")
    
    # Visualize prediction results
    with torch.no_grad():
        all_inputs = []
        all_targets = []
        all_predictions = []
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            preds = model(inputs)
            all_inputs.append(inputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_predictions.append(preds.cpu().numpy())
        
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
        plt.savefig(f'figure/correlations_T{t}.png')
        plt.show()

    # Save the final model
    final_model_path = 'model_pth/final_correlation_model.pth'
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"Final model saved as '{final_model_path}'")
    logger.info(f"Best model saved as '{checkpoint_path}'")

if __name__ == "__main__":
    main()
