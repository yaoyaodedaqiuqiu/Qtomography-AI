import os
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import CorrelationDatasetPyTorch
from model import CorrelationModel
from torch.utils.tensorboard import SummaryWriter
import time
from torchviz import make_dot

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')
logger = logging.getLogger()

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

def train_model(model, train_loader, criterion, optimizer, num_epochs, dataset_size, checkpoint_path, writer):
    """
    Train the model with the given parameters
    
    Parameters:
        model: The model to train
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer for training
        num_epochs: Number of epochs to train
        dataset_size: Size of the dataset
        checkpoint_path: Path to save the best model
        writer: TensorBoard SummaryWriter
        
    Returns:
        best_loss: The best loss achieved during training
    """
    best_loss = float('inf')
    
    # Ensure the checkpoint directory exists
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    logger.info("Starting training...")
    model.train()
    
    # Get a batch of data for model visualization
    sample_inputs, sample_targets = next(iter(train_loader))
    sample_inputs = sample_inputs.to(device)
    
    # Add model graph to TensorBoard
    writer.add_graph(model, sample_inputs)
    
    # Training loop
    global_step = 0
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        batch_count = 0
        
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            # Log batch loss
            writer.add_scalar('Loss/batch', loss.item(), global_step)
            
            # Log parameter histograms (every 50 steps)
            if global_step % 50 == 0:
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        writer.add_histogram(f"Parameters/{name}", param.data, global_step)
                        if param.grad is not None:
                            writer.add_histogram(f"Gradients/{name}", param.grad, global_step)
            
            epoch_loss += loss.item() * inputs.size(0)
            batch_count += 1
            global_step += 1
            
            # Log predictions vs targets for a few samples (every 100 steps)
            if global_step % 100 == 0:
                with torch.no_grad():
                    for i in range(min(5, len(inputs))):
                        fig_name = f"pred_vs_target_step{global_step}_sample{i}"
                        fig = plot_prediction_vs_target(outputs[i].cpu().numpy(), targets[i].cpu().numpy())
                        writer.add_figure(fig_name, fig, global_step)
        
        epoch_loss /= dataset_size
        
        # Log epoch loss
        writer.add_scalar('Loss/epoch', epoch_loss, epoch)
        
        # Log learning rate
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Learning_rate', current_lr, epoch)
        
        # Calculate elapsed time and ETA
        elapsed_time = time.time() - start_time
        eta = elapsed_time / (epoch + 1) * (num_epochs - epoch - 1)
        
        if (epoch + 1) % 50 == 0 or epoch == 0:
            logger.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.6f}, "
                       f"Elapsed: {format_time(elapsed_time)}, ETA: {format_time(eta)}")
        
        # Check if it's the best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"Best model updated at epoch {epoch+1} with loss {best_loss:.6f}")
            
            # Visualize model architecture for the best model
            if epoch == 0:  # Only do this once to save computation
                try:
                    # Create a visual representation of the model
                    y = model(sample_inputs)
                    dot = make_dot(y, params=dict(model.named_parameters()))
                    dot.format = 'png'
                    dot.render(os.path.join(writer.log_dir, 'model_architecture'))
                    logger.info(f"Model architecture visualization saved to {writer.log_dir}")
                except Exception as e:
                    logger.warning(f"Failed to visualize model architecture: {e}")
    
    # Log final model parameters
    for name, param in model.named_parameters():
        if param.requires_grad:
            writer.add_histogram(f"Final_Parameters/{name}", param.data, 0)
    
    logger.info("Training completed.")
    return best_loss

def format_time(seconds):
    """Format time in seconds to a readable string"""
    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def plot_prediction_vs_target(prediction, target):
    """Create a matplotlib figure comparing prediction and target"""
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(prediction))
    ax.plot(x, prediction, 'b-', label='Prediction')
    ax.plot(x, target, 'r-', label='Target')
    ax.set_xlabel('Index')
    ax.set_ylabel('Value')
    ax.set_title('Prediction vs Target')
    ax.legend()
    
    return fig

def main(model_type='mlp'):
    # Parameter settings
    N = 9
    
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
    
    # Initialize model based on model_type
    output_dim = N - 1  # Number of correlations
    input_dim = 3  # J, h, T
    hidden_dim = 128
    
    # Setup TensorBoard
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join('runs', f'{model_type}_{timestamp}')
    writer = SummaryWriter(log_dir=log_dir)
    logger.info(f"TensorBoard logs will be saved to {log_dir}")
    
    if model_type.lower() == 'mlp':
        from model import CorrelationModel
        model = CorrelationModel(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
    elif model_type.lower() == 'transformer':
        # Import and initialize transformer model when implemented
        logger.info("Transformer model will be implemented in the future.")
        from model import CorrelationModel  # Fallback to MLP for now
        model = CorrelationModel(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
    elif model_type.lower() == 'cnn':
        # Import and initialize CNN model when implemented
        logger.info("CNN model will be implemented in the future.")
        from model import CorrelationModel  # Fallback to MLP for now
        model = CorrelationModel(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
    elif model_type.lower() == 'gcn':
        # Import and initialize GCN model when implemented
        logger.info("GCN model will be implemented in the future.")
        from model import CorrelationModel  # Fallback to MLP for now
        model = CorrelationModel(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
    else:
        logger.warning(f"Unknown model type: {model_type}. Using default MLP model.")
        from model import CorrelationModel
        model = CorrelationModel(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
    
    model = model.to(device)
    logger.info(f"Using model type: {model_type}")
    
    # Log model hyperparameters
    writer.add_hparams(
        {
            'model_type': model_type,
            'input_dim': input_dim,
            'hidden_dim': hidden_dim,
            'output_dim': output_dim,
            'batch_size': batch_size,
            'learning_rate': 1e-3,
            'num_epochs': 500
        },
        {}  # Metrics dict (will be filled during training)
    )
    
    # Initialize loss function and optimizer
    criterion = nn.MSELoss()
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Model checkpoint path
    model_dir = 'model_pth'
    os.makedirs(model_dir, exist_ok=True)
    checkpoint_path = os.path.join(model_dir, f'best_{model_type}_correlation_model.pth')
    
    # Train the model
    num_epochs = 500
    best_loss = train_model(
        model=model,
        train_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=num_epochs,
        dataset_size=len(dataset),
        checkpoint_path=checkpoint_path,
        writer=writer
    )
    
    # Save the final model
    final_model_path = os.path.join(model_dir, f'final_{model_type}_correlation_model.pth')
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"Final model saved as '{final_model_path}'")
    logger.info(f"Best model saved as '{checkpoint_path}' with loss {best_loss:.6f}")
    
    # Add final metrics to hparams
    writer.add_hparams(
        {},  # Already added above
        {'hparam/best_loss': best_loss}
    )
    
    # Close TensorBoard writer
    writer.close()
    
    # Print instructions for viewing TensorBoard
    logger.info("\nTo view training visualizations, run:")
    logger.info(f"tensorboard --logdir={os.path.dirname(log_dir)}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train correlation models with different architectures')
    parser.add_argument('--model', type=str, default='mlp', choices=['mlp', 'transformer', 'cnn', 'gcn'],
                        help='Model architecture to use (default: mlp)')
    
    args = parser.parse_args()
    main(model_type=args.model)
