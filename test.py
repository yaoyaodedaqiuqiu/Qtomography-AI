import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import logging
from dataset import CorrelationDatasetPyTorch, generate_and_save_data  # 假设 `generate_and_save_data` 已定义
from model import CorrelationModel, PositionAwareModel, TransformerModel
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression

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
    elif model_type == "Transformer":
        return TransformerModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def plot_correlation_comparison(targets, preds, figure_folder, model_type):
    """Plot scatter comparison with linear fit"""
    plt.figure(figsize=(15, 10))
    for i in range(preds.shape[1]):
        plt.subplot(3, 3, i+1)
        # Scatter plot
        plt.scatter(targets[:, i], preds[:, i], alpha=0.3, label="Predictions")
        
        # Linear regression fit line
        reg = LinearRegression().fit(targets[:, i].reshape(-1, 1), preds[:, i])
        plt.plot(targets[:, i], reg.predict(targets[:, i].reshape(-1, 1)), color='r', label='Fit Line')

        # Labels and title
        plt.xlabel('True')
        plt.ylabel('Predicted')
        plt.title(f'Correlation d={i+1}')
        plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figure_folder, f'{model_type}_correlation_comparison_with_fit.png'))
    plt.close()

def plot_error_heatmap(targets, preds, figure_folder, model_type):
    """Plot error heatmap"""
    errors = np.abs(preds - targets)

    plt.figure(figsize=(10, 6))
    plt.imshow(errors, cmap='plasma', aspect='auto')
    plt.colorbar(label='Prediction Error')
    plt.xlabel('Distance (d)')
    plt.ylabel('Sample Index')
    plt.title(f'{model_type} Error Heatmap')
    plt.savefig(os.path.join(figure_folder, f'{model_type}_error_heatmap.png'))
    plt.close()

def plot_error_distribution(targets, preds, figure_folder, model_type):
    """Plot prediction error distribution"""
    errors = np.abs(preds - targets)
    
    plt.figure(figsize=(8, 6))
    plt.hist(errors.flatten(), bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title(f'{model_type} Prediction Error Distribution')
    plt.xlabel('Error')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(os.path.join(figure_folder, f'{model_type}_error_distribution.png'))
    plt.close()

def plot_performance_metrics(targets, preds, figure_folder, model_type):
    """Plot various performance metrics"""
    mse_values = mean_squared_error(targets, preds, multioutput='raw_values')
    mae_values = mean_absolute_error(targets, preds, multioutput='raw_values')
    r2_values = r2_score(targets, preds, multioutput='raw_values')
    
    # Create subplots for each metric
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    axs[0].bar(range(1, len(mse_values) + 1), mse_values)
    axs[0].set_title(f'{model_type} MSE per Distance')
    axs[0].set_xlabel('Distance (d)')
    axs[0].set_ylabel('MSE')

    axs[1].bar(range(1, len(mae_values) + 1), mae_values)
    axs[1].set_title(f'{model_type} MAE per Distance')
    axs[1].set_xlabel('Distance (d)')
    axs[1].set_ylabel('MAE')

    axs[2].bar(range(1, len(r2_values) + 1), r2_values)
    axs[2].set_title(f'{model_type} R² per Distance')
    axs[2].set_xlabel('Distance (d)')
    axs[2].set_ylabel('R²')

    plt.tight_layout()
    plt.savefig(os.path.join(figure_folder, f'{model_type}_performance_metrics.png'))
    plt.close()

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Test correlation prediction model')
    parser.add_argument('--model', type=str, default='MLP', choices=['MLP', 'PositionAware', 'Transformer'],
                        help='Model type: MLP, PositionAware, or Transformer')
    parser.add_argument('--model_type', type=str, default='Cluster', choices=['Ising', 'Heisenberg', 'Cluster'],
                        help='Choose the physical model: Ising, Heisenberg, or Cluster')
    parser.add_argument('--data_folder', type=str, default='test_dataset', help='Path to dataset folder')
    parser.add_argument('--N', type=int, default=9, help='Number of qubits')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint. If not provided, it will be constructed automatically.')
    args = parser.parse_args()

    figure_folder = "figure_test"
    os.makedirs(figure_folder, exist_ok=True)
    # Dataset configuration
    N = 9
    J_list = np.linspace(2.0, 4.0, 11)
    h_list = np.linspace(1.0, 2.0, 11)
    t_list = np.linspace(2.0, 4.0, 11)
    # Data generation
    logger.info("Generating and saving test data...")
    # Generate and save data for the selected model
    # generate_and_save_data(N, J_list, h_list, t_list, model_type=args.model_type, data_folder='test_dataset')
    logger.info("Test data generation completed.")

    # Dataset configuration
    N = args.N

    # Load dataset based on the selected physical model
    test_dataset = CorrelationDatasetPyTorch(data_folder=args.data_folder, N=N, model_type=args.model_type)
    if len(test_dataset) == 0:
        logger.error("Test dataset is empty")
        return

    # Initialize model
    sample_input, _ = test_dataset[0]
    model = create_model(
        model_type=args.model,
        input_dim=len(sample_input),
        hidden_dim=128,  # Use default hidden_dim or extract from checkpoint if saved
        output_dim=N - 1
    ).to(device)

    # Load model checkpoint
    if args.checkpoint is None:
        args.checkpoint = f"model_pth/best_model_{args.model}_for_{args.model_type}.pth"
    
    if not os.path.exists(args.checkpoint):
        logger.error(f"Model checkpoint not found: {args.checkpoint}")
        return

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    logger.info(f"Loaded {args.model} model with parameters: {sum(p.numel() for p in model.parameters())}")

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

    print(f"Metrics for {args.model}:")
    print(f"  MSE: {mse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  R²: {r2:.4f}")

    # Distance-dependent error
    distance_errors = []
    for d in range(preds.shape[1]):
        distance_errors.append(mean_squared_error(targets[:, d], preds[:, d]))
    print(f"  Distance-dependent MSE: {distance_errors}")

    # Generate enhanced visualizations
    logger.info("Generating enhanced visualizations...")
    plot_correlation_comparison(targets, preds, figure_folder, args.model)
    plot_error_heatmap(targets, preds, figure_folder, args.model)
    plot_error_distribution(targets, preds, figure_folder, args.model)
    plot_performance_metrics(targets, preds, figure_folder, args.model)

    # Save predictions
    np.savez(os.path.join(figure_folder, f'{args.model}_predictions.npz'),
             inputs=inputs, targets=targets, preds=preds)

    logger.info(f"Testing completed. Results saved in {figure_folder}")

if __name__ == "__main__":
    main()
