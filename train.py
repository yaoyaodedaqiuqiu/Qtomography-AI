# train.py
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter  # Added TensorBoard support
from dataset import CorrelationDatasetPyTorch
from model import CorrelationModel, PositionAwareModel
import matplotlib.pyplot as plt
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')
logger = logging.getLogger()

# Configuration parameters
parser = argparse.ArgumentParser(description='Train correlation prediction model')
parser.add_argument('--model', type=str, default='MLP', choices=['MLP', 'PositionAware'],
                    help='Model type: MLP or PositionAware')
parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden layer dimension')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--epochs', type=int, default=500, help='Number of epochs')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--log_dir', type=str, default='runs/exp1',
                    help='TensorBoard log directory')
args = parser.parse_args()

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

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
    return total_loss / len(dataloader.dataset)

def main():
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize TensorBoard
    writer = SummaryWriter(log_dir=args.log_dir)
    
    # Dataset configuration
    N = 9
    J_list = np.linspace(1.0, 2.0, 11)
    h_list = np.linspace(0.0, 1.0, 11)
    t_list = np.linspace(1.0, 2.0, 11)
    
    # Data generation
    logger.info("Generating and saving test data...")
    # generate_and_save_data(N, J_list, h_list, t_list, data_folder='correlation_dataset')
    logger.info("Test data generation completed.")

    output_dim = N - 1
    
    # Load dataset
    data_folder = "correlation_dataset"
    full_dataset = CorrelationDatasetPyTorch(data_folder, N)
    sample_input, sample_target = full_dataset[0]
    input_dim = sample_input.shape[0]
    
    # Create model
    model = create_model(
        model_type=args.model,
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=output_dim
    ).to(device)
    
    # Record model architecture
    dummy_input = torch.randn(1, input_dim).to(device)
    writer.add_graph(model, dummy_input)
    
    # Define training components
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Data loader
    dataloader = DataLoader(full_dataset, batch_size=args.batch_size, shuffle=True)
    
    # Training loop
    best_loss = float('inf')
    print(f"Start training {args.model} model...")
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, dataloader, criterion, optimizer, device)
        
        # Log training metrics
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)
        
        # Print progress
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1:03d}/{args.epochs} | Loss: {train_loss:.4e}")
        
        # Save best model
        if train_loss < best_loss:
            best_loss = train_loss
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'loss': train_loss,
            }, f"model_pth/best_model_{args.model}.pth")
    
    # Log hyperparameters and final metrics
    writer.add_hparams(
        {
            'model': args.model,
            'lr': args.lr,
            'batch_size': args.batch_size,
            'hidden_dim': args.hidden_dim
        },
        {
            'final_loss': best_loss
        }
    )
    
    writer.close()
    print(f"Training completed. Best loss: {best_loss:.4e}")

if __name__ == "__main__":
    main()
