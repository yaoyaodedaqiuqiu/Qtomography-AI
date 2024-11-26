# train.py
import torch
import torch.optim as optim
import torch.utils.data as data
import pickle
import numpy as np
from model import GibbsStateModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Load Gibbs states from pickle file
def load_gibbs_states(filename='gibbs_states.pkl'):
    with open(filename, 'rb') as f:
        gibbs_states = pickle.load(f)
    return gibbs_states

# Preprocess Gibbs states into training data

def preprocess_data(gibbs_states):
    inputs = []
    targets = []
    
    # For each temperature (T) and its corresponding Gibbs state matrix
    for T, state in gibbs_states.items():
        # Flatten the real and imaginary parts of the Gibbs state matrix separately
        real_part = state.real.flatten()
        imag_part = state.imag.flatten()
        
        # Concatenate real and imaginary parts
        flattened_state = np.concatenate([real_part, imag_part])
        
        # Use temperature as target
        inputs.append(flattened_state)
        targets.append(T)
    
    # Convert to numpy arrays
    inputs = np.array(inputs)
    targets = np.array(targets)
    
    # Normalize the inputs (separate for real and imaginary parts)
    scaler = MinMaxScaler()
    inputs = scaler.fit_transform(inputs)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(inputs, targets, test_size=0.2, random_state=42)
    
    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
    
    return X_train, X_test, y_train, y_test, scaler


# Training loop
def train(model, X_train, y_train, X_test, y_test, num_epochs=100, batch_size=32, learning_rate=1e-3):
    # Define loss function and optimizer
    criterion = torch.nn.MSELoss()  # Mean Squared Error for regression
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Create data loader
    train_data = data.TensorDataset(X_train, y_train)
    train_loader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, targets in train_loader:
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            
            # Compute loss
            loss = criterion(outputs, targets)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # Print statistics
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}")
        
        # Evaluate on the test set
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            test_loss = criterion(test_outputs, y_test)
            print(f"Test Loss after Epoch {epoch+1}: {test_loss.item()}")
    
    print("Training complete.")

# Save the trained model
def save_model(model, filename='gibbs_state_model.pth'):
    torch.save(model.state_dict(), filename)
    print(f"Model saved to {filename}")

# Main function to load data, train, and save the model
def main():
    gibbs_states = load_gibbs_states()  # Load data
    X_train, X_test, y_train, y_test, scaler = preprocess_data(gibbs_states)
    
    input_size = X_train.shape[1]  # The size of the flattened Gibbs state (2^n * 2^n)
    
    # Initialize the model
    model = GibbsStateModel(input_size=input_size, hidden_size=128, output_size=1)
    
    # Train the model
    train(model, X_train, y_train, X_test, y_test, num_epochs=100)
    
    # Save the trained model
    save_model(model)

if __name__ == '__main__':
    main()
