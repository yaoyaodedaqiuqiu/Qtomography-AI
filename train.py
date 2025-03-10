# train.py
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import logging
from dataset import generate_and_save_data, CorrelationDatasetPyTorch
from model import CorrelationModel  

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')
logger = logging.getLogger()

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Define PyTorch Dataset class for loading h, j, t and corresponding correlation terms

def main():
    # Parameter settings
    N = 9
    J_list = np.linspace(1.0, 2.0, 11)      # Example j values
    h_list = np.linspace(0, 1, 11)    # 21 h values from 0 to 1 with a step size of 0.05
    t_list = np.linspace(1.0, 2.0, 11)        # Example temperature values
    
    # Data generation (comment out after the first run)
    logger.info("Generating and saving data...")
    # generate_and_save_data(N, J_list, h_list, t_list, data_folder='correlation_dataset')
    logger.info("Data generation completed.")
    
    # Create dataset
    data_folder = 'correlation_dataset'
    dataset = CorrelationDatasetPyTorch(data_folder, N)
    if len(dataset) == 0:
        logger.error("No data found in the dataset. Please check the folder path and file format.")
        return

    logger.info(f"Dataset size: {len(dataset)}")
    
    # Create data loader
    batch_size = 32
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model, loss function, and optimizer
    output_dim = N - 1  # Number of correlations
    model = CorrelationModel(input_dim=3, hidden_dim=128, output_dim=output_dim).to(device)
    criterion = nn.MSELoss()
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Model checkpoint path
    checkpoint_path = 'model_pth/best_correlation_model.pth'
    best_loss = float('inf')
    
    # Ensure the checkpoint directory exists
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
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
        
        epoch_loss /= len(dataset)
        if (epoch + 1) % 50 == 0 or epoch == 0:
            logger.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.6f}")
        
        # Check if it's the best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"Best model updated at epoch {epoch+1} with loss {best_loss:.6f}")
    
    logger.info("Training completed.")
    
    # Save the final model
    final_model_path = 'model_pth/final_correlation_model.pth'
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"Final model saved as '{final_model_path}'")
    logger.info(f"Best model saved as '{checkpoint_path}'")

if __name__ == "__main__":
    main()
