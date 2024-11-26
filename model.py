# model.py
import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple feedforward neural network for regression
class GibbsStateModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, output_size=1):
        """
        Args:
            input_size: The size of the flattened Gibbs state (for a 2^n x 2^n matrix).
            hidden_size: The size of the hidden layers.
            output_size: The output size (temperature, scalar value).
        """
        super(GibbsStateModel, self).__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor (Gibbs state flattened into a vector).
            
        Returns:
            output: Predicted temperature.
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        output = self.fc3(x)
        return output
