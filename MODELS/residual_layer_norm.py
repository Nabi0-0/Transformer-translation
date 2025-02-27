#for Implement Layer Normalization & Residual Connection

import torch
import torch.nn as nn

#layer normalization

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))  # Scale parameter
        self.beta = nn.Parameter(torch.zeros(d_model))  # Shift parameter
        self.eps = eps  # Small constant to prevent division by zero

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

#resedual connection flow

class ResidualConnection(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(ResidualConnection, self).__init__()
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))  # Apply layer norm, sublayer, dropout, and residual connection
