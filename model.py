import torch
import torch.nn as nn
class TransformerModel(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=128, output_dim=8, num_heads=4, num_layers=2):
        super(TransformerModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Transformer Encoder
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 2
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=num_layers
        )
        
        # Input Linear Layer (to match the input dimensions)
        self.input_fc = nn.Linear(input_dim, hidden_dim)
        
        # Output Linear Layer (to match the output dimensions)
        self.output_fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x shape: [batch_size, input_dim] => [batch_size, 1, input_dim]
        x = x.unsqueeze(1)  # Add sequence length dimension (seq_len=1)
        
        # Convert shape to [seq_len, batch_size, hidden_dim] (for Transformer)
        x = self.input_fc(x)  # Now shape is [batch_size, 1, hidden_dim] => [1, batch_size, hidden_dim]
        
        # Pass through Transformer (requires [seq_len, batch_size, hidden_dim])
        x = x.permute(1, 0, 2)  # Now shape is [1, batch_size, hidden_dim]
        
        # Pass through Transformer
        x = self.transformer_encoder(x)
        
        # Take the output corresponding to the last sequence element
        x = x[-1, :, :]  # [batch_size, hidden_dim]
        
        # Final output layer
        x = self.output_fc(x)
        return x

    
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
