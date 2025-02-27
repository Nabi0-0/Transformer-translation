#feed-forward layer

import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)  # First linear transformation
        self.relu = nn.ReLU()  # Non-linearity
        self.dropout = nn.Dropout(dropout)  # If this line exists, dropout is required
        self.fc2 = nn.Linear(d_ff, d_model)  # Second linear transformation

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))  # Apply FFN