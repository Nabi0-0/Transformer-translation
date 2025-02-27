#feed-forward layer

import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)  # First linear transformation
        self.relu = nn.ReLU()  # Non-linearity
        self.fc2 = nn.Linear(d_ff, d_model)  # Second linear transformation

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))  # Apply FFN

# Test the module
if __name__ == "__main__":
    batch_size = 2
    seq_length = 5
    d_model = 512
    d_ff = 2048  # Commonly set as 4 * d_model

    ffn = FeedForward(d_model, d_ff)
    x = torch.randn(batch_size, seq_length, d_model)
    output = ffn(x)

    print("Output Shape:", output.shape)  # Should be (batch_size, seq_length, d_model)
