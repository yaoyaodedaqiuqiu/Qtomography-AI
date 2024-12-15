import torch.nn as nn

class CorrelationModel(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=128, output_dim=8):
        super(CorrelationModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)
