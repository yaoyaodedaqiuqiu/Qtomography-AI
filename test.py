# test.py
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import logging
from dataset import CorrelationDatasetPyTorch, generate_and_save_data  # Add data generation import
from model import CorrelationModel, PositionAwareModel
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')
logger = logging.getLogger()

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_model(model_type, input_dim, hidden_dim, output_dim):
    """Model factory function"""
    if model_type == "PositionAware":
        return PositionAwareModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            max_d=output_dim
        )
    elif model_type == "MLP":
        return CorrelationModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Test correlation prediction model')
    parser.add_argument('--model_type', type=str, required=True, choices=['MLP', 'PositionAware'],
                       help='Model type to test: MLP or PositionAware')
    parser.add_argument('--checkpoint', type=str, default='model_pth/best_model.pth',
                       help='Path to model checkpoint')
    args = parser.parse_args()

    # Configure paths
    args.checkpoint = f"model_pth/best_model_{args.model_type}.pth"
    figure_folder = "figure_test"
    os.makedirs(figure_folder, exist_ok=True)


    # Dataset configuration
    N = 9
    J_list = np.linspace(2.0, 4.0, 11)
    h_list = np.linspace(1.0, 2.0, 11)
    t_list = np.linspace(2.0, 4.0, 11)
    
    # Generate test data with same logic
    logger.info("Generating and saving test data...")
    # generate_and_save_data(N, J_list, h_list, t_list, data_folder='test_dataset')
    logger.info("Test data generation completed.")

    # Load checkpoint
    if not os.path.exists(args.checkpoint):
        logger.error(f"Model checkpoint not found: {args.checkpoint}")
        return

    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Load test data
    test_dataset = CorrelationDatasetPyTorch("test_dataset", N=9)
    if len(test_dataset) == 0:
        logger.error("Test dataset is empty")
        return
    
    # Initialize model
    sample_input, _ = test_dataset[0]
    model = create_model(
        model_type=args.model_type,
        input_dim=len(sample_input),
        hidden_dim=checkpoint.get('hidden_dim', 128),
        output_dim=checkpoint.get('output_dim', 8)
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    logger.info(f"Loaded {args.model_type} model with parameters: {sum(p.numel() for p in model.parameters())}")

    # Create data loader
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Evaluate model
    all_preds, all_targets, all_inputs = [], [], []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            all_inputs.append(inputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_preds.append(outputs.cpu().numpy())

    # Combine data
    inputs = np.vstack(all_inputs)
    targets = np.vstack(all_targets)
    preds = np.vstack(all_preds)

    # Calculate metrics
    mse = mean_squared_error(targets, preds)
    mae = mean_absolute_error(targets, preds)
    r2 = r2_score(targets, preds)

    print(f"Metrics for {args.model_type}:")
    print(f"  MSE: {mse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  R²: {r2:.4f}")

    # Distance-dependent error
    distance_errors = []
    for d in range(preds.shape[1]):
        distance_errors.append(mean_squared_error(targets[:, d], preds[:, d]))
    print(f"  Distance-dependent MSE: {distance_errors}")

    # Plot distance-dependent error
    plt.figure()
    plt.plot(range(1, len(distance_errors) + 1), distance_errors)
    plt.xlabel("Distance (d)")
    plt.ylabel("MSE")
    plt.title(f"{args.model_type} - Distance-dependent Error")
    plt.savefig(os.path.join(figure_folder, f'{args.model_type}_distance_error.png'))
    plt.close()
    # Visualization functions
    def plot_correlation_comparison():
        plt.figure(figsize=(15, 10))
        for i in range(preds.shape[1]):
            plt.subplot(3, 3, i+1)
            plt.scatter(targets[:, i], preds[:, i], alpha=0.3)
            plt.plot([-1, 1], [-1, 1], 'r--')
            plt.xlabel('True')
            plt.ylabel('Predicted')
            plt.title(f'Correlation d={i+1}')
        plt.tight_layout()
        plt.savefig(os.path.join(figure_folder, f'{args.model_type}_correlation_comparison.png'))
        plt.close()

    def plot_heatmap():
        h = inputs[:, 0]
        t = inputs[:, 2]
        
        plt.figure(figsize=(10, 6))
        plt.hexbin(h, t, C=preds[:, 0], gridsize=20, cmap='plasma')
        plt.colorbar(label='Correlation Strength')
        plt.xlabel('Magnetic Field (h)')
        plt.ylabel('Temperature (T)')
        plt.title(f'{args.model_type} Correlation Heatmap')
        plt.savefig(os.path.join(figure_folder, f'{args.model_type}_heatmap.png'))
        plt.close()

    # Generate plots
    logger.info("Generating visualizations...")
    plot_correlation_comparison()
    plot_heatmap()

    # Save predictions
    np.savez(os.path.join(figure_folder, f'{args.model_type}_predictions.npz'),
             inputs=inputs, targets=targets, preds=preds)

    logger.info(f"Testing completed. Results saved in {figure_folder}")

if __name__ == "__main__":
    main()
