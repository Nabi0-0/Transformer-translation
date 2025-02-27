#test feedforward network 

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from models.feed_forward import FeedForward

def test_feedforward():
    batch_size = 2
    seq_length = 5
    d_model = 512
    d_ff = 2048  # Hidden layer size

    ffn = FeedForward(d_model, d_ff)
    
    x = torch.randn(batch_size, seq_length, d_model)
    output = ffn(x)

    assert output.shape == (batch_size, seq_length, d_model), "Output shape is incorrect!"

    print("✅ Feedforward Network Test Passed!")

if __name__ == "__main__":
    test_feedforward()
