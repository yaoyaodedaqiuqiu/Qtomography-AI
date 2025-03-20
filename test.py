# test.py
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import logging
from dataset import generate_and_save_data, IsingHamiltonian,CorrelationDatasetPyTorch, Z, I, kron_n
from model import CorrelationModel

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')
logger = logging.getLogger()

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")


def main():
    # Parameter settings for test data
    N = 9
    J_list = np.linspace(1.5, 2.5, 11)           # Same j values as training
    h_list = np.linspace(0.5, 1.5, 11)        # 6 h values from 0 to 1 with a step size of 0.2
    t_list = np.linspace(1.5, 2.5, 11)            # New temperature values for testing
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
    
    # Get unique h and T values
    unique_h = np.unique(h_vals)
    unique_t = np.unique(t_vals)
    correlation_names = [f'Correlation {i+2}' for i in range(output_dim)]
    
    # Create a directory for figures
    figure_test_folder = 'figure_test'
    os.makedirs(figure_test_folder, exist_ok=True)
    
    # =======================
    # 1. Heatmap of Correlation Term
    # =======================
    logger.info("Generating heatmap of correlation terms...")
    heatmap_data = np.zeros((len(unique_t), len(unique_h)))
    
    for i, T in enumerate(unique_t):
        for j, h in enumerate(unique_h):
            mask = (t_vals == T) & (h_vals == h)
            if not np.any(mask):
                continue
            # Considering multiple samples per (T, h), take average
            heatmap_data[i, j] = np.mean(all_predictions[mask, 0])  # Correlation <σ_1^z σ_2^z>
    
    plt.figure(figsize=(12 * (1/2.54), 9 * (1/2.54)))  # 12x9 cm
    plt.imshow(heatmap_data, extent=[min(unique_h), max(unique_h), min(unique_t), max(unique_t)],
               origin='lower', aspect='auto', cmap="plasma")
    plt.colorbar(label='Predicted Correlation <σ₁ᶻ σ₂ᶻ>')
    plt.title(r"Heatmap of Predicted Correlation <σ₁ᶻ σ₂ᶻ> Across T and h")
    plt.xlabel("h")
    plt.ylabel("Temperature T")
    plt.tight_layout()
    heatmap_path = os.path.join(figure_test_folder, 'heatmap_correlation.png')
    plt.savefig(heatmap_path)
    logger.info(f"Saved heatmap: {heatmap_path}")
    plt.close()
    
    # =======================
    # 2. Line Plots for Specific h Values
    # =======================

    critical_h = 0.5
    logger.info("Analyzing critical point h=0.5...")
    plt.figure(figsize=(12 * (1/2.54), 9 * (1/2.54)))  # 12x9 cm
    for T in unique_t:
        mask = (t_vals == T) & (h_vals == critical_h)
        if not np.any(mask):
            continue
        predicted_corr = all_predictions[mask]
        true_corr = all_targets[mask]
        plt.plot(range(2, N + 1), predicted_corr[0], label=f'Predicted T={T}')
        plt.plot(range(2, N + 1), true_corr[0], linestyle='--', label=f'True T={T}')
    
    plt.xlabel('j')
    plt.ylabel(r'$\langle \sigma_1^z \sigma_j^z \rangle$')
    plt.title(f'Correlation Functions at Critical Point h={critical_h}')
    plt.grid(True)
    plt.tight_layout()
    critical_plot_path = os.path.join(figure_test_folder, f'correlation_functions_h{critical_h}.png')
    plt.savefig(critical_plot_path)
    logger.info(f"Saved critical point analysis plot: {critical_plot_path}")
    plt.close()

    # =======================
    # 3. Correlation vs Temperature at Critical Point (h=0.5)
    # =======================
    critical_h = 0.5
    logger.info("Generating correlation vs Temperature plot at critical point h=0.5...")
    plt.figure(figsize=(12 * (1/2.54), 9 * (1/2.54)))  # 12x9 cm
    
    for j in range(2, N + 1):
        correlations_T = []
        for T in unique_t:
            mask = (t_vals == T) & (h_vals == critical_h)
            if not np.any(mask):
                correlations_T.append(np.nan)
                continue
            predicted_corr = all_predictions[mask, j-2]  # j starts from 2
            correlations_T.append(np.mean(predicted_corr))
        
        plt.plot(unique_t, correlations_T, marker='o', label=f'j={j}')
    
    plt.xlabel('Temperature T')
    plt.ylabel(r'$\langle \sigma_1^z \sigma_j^z \rangle$')
    plt.title(f'Correlation <σ₁ᶻ σ_jᶻ> vs Temperature at Critical Point h={critical_h}')
    plt.legend(title='j Values')
    plt.grid(True)
    plt.tight_layout()
    critical_T_plot_path = os.path.join(figure_test_folder, f'correlation_vs_T_h{critical_h}.png')
    plt.savefig(critical_T_plot_path)
    logger.info(f"Saved correlation vs T plot at h={critical_h}: {critical_T_plot_path}")
    plt.close()
    
    
    # =======================
    # 4. Property Precision vs Sample Size
    # =======================
    # Assuming multiple samples per (T, h), we can analyze how the correlation precision improves
    # with the number of samples. Here, we'll plot the standard deviation of predictions
    # as a proxy for precision.
    
    logger.info("Analyzing correlation precision vs sample size...")
    precision_data = {}
    for h in unique_h:
        for T in unique_t:
            mask = (t_vals == T) & (h_vals == h)
            if not np.any(mask):
                continue
            predicted_corr = all_predictions[mask, 0]
            precision = np.std(predicted_corr)
            precision_data[(T, h)] = precision
    
    # Convert precision_data to 2D array for heatmap
    precision_heatmap = np.zeros((len(unique_t), len(unique_h)))
    for i, T in enumerate(unique_t):
        for j, h in enumerate(unique_h):
            if (T, h) in precision_data:
                precision_heatmap[i, j] = precision_data[(T, h)]
            else:
                precision_heatmap[i, j] = np.nan  # or some default value
    
    plt.figure(figsize=(12 * (1/2.54), 9 * (1/2.54)))  # 12x9 cm
    plt.imshow(precision_heatmap, extent=[min(unique_h), max(unique_h), min(unique_t), max(unique_t)],
               origin='lower', aspect='auto', cmap="viridis")
    plt.colorbar(label='Standard Deviation of Predicted Correlation')
    plt.title("Correlation Precision Across T and h")
    plt.xlabel("h")
    plt.ylabel("Temperature T")
    plt.tight_layout()
    precision_heatmap_path = os.path.join(figure_test_folder, 'precision_heatmap.png')
    plt.savefig(precision_heatmap_path)
    logger.info(f"Saved precision heatmap: {precision_heatmap_path}")
    plt.close()
    
    logger.info("Testing and plotting completed.")

if __name__ == "__main__":
    main()
