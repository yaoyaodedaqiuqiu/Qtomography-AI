import torch
import torch.nn as nn

class CorrelationModel(nn.Module):
    """Baseline MLP model"""
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

class PositionAwareModel(nn.Module):
    """Physics-informed model with positional awareness"""
    def __init__(self, input_dim=3, hidden_dim=128, max_d=8):
        super().__init__()
        self.max_d = max_d  # Maximum spin distance (d ranges 1-8 when N=9)
        
        # Shared feature extraction layers
        self.shared_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Position-aware prediction head (weight sharing)
        self.position_head = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),  # +1 for distance encoding
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        batch_size = x.shape[0]
        shared_features = self.shared_net(x)
        d = torch.arange(1, self.max_d+1, dtype=torch.float32, device=x.device)
        d = d.view(1, -1).repeat(batch_size, 1)
        shared_expanded = shared_features.unsqueeze(1).repeat(1, self.max_d, 1)
        combined = torch.cat([shared_expanded, d.unsqueeze(-1)], dim=-1)
        return self.position_head(combined).squeeze(-1)
