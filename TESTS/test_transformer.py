#for testing transformer model

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from models.Transformer import Transformer

def test_transformer():
    input_vocab_size = 10000
    target_vocab_size = 10000
    d_model = 512
    num_heads = 8
    d_ff = 2048
    num_layers = 6
    dropout = 0.1

    transformer = Transformer(input_vocab_size, target_vocab_size, d_model, num_heads, d_ff, num_layers, dropout)
    
    src = torch.randint(0, input_vocab_size, (2, 10))  # (batch_size, seq_len)
    trg = torch.randint(0, target_vocab_size, (2, 10))  # (batch_size, seq_len)
    src_mask = torch.ones((2, 1, 1, 10))  # Dummy mask (for testing)
    trg_mask = torch.ones((2, 1, 10, 10))  # Dummy mask (for testing)

    output = transformer(src, trg, src_mask, trg_mask)
    assert output.shape == (2, 10, target_vocab_size), f"Unexpected shape: {output.shape}"

    print("✅ Transformer model test passed!")

if __name__ == "__main__":
    test_transformer()
