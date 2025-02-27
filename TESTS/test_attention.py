#for testing mult head attention mechanism


import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from models.multi_head_attention import MultiHeadAttention

def test_multi_head_attention():
    batch_size = 2
    seq_length = 5
    d_model = 512
    num_heads = 8

    mha = MultiHeadAttention(d_model, num_heads)
    
    # Random input tensors (batch_size, seq_length, d_model)
    queries = torch.randn(batch_size, seq_length, d_model)
    keys = torch.randn(batch_size, seq_length, d_model)
    values = torch.randn(batch_size, seq_length, d_model)

    # Forward pass
    output, attn_weights = mha(queries, keys, values)

    # Assertions
    assert output.shape == (batch_size, seq_length, d_model), "Output shape is incorrect!"
    assert attn_weights.shape == (batch_size, num_heads, seq_length, seq_length), "Attention shape is incorrect!"

    print("✅ Multi-Head Attention Test Passed!")

if __name__ == "__main__":
    test_multi_head_attention()
