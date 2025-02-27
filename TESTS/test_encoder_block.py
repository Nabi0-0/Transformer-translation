#for testing encoder

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import torch
from models.encoder import EncoderBlock

def test_encoder_block():
    batch_size = 2
    seq_length = 5
    d_model = 512
    num_heads = 8
    d_ff = 2048

    x = torch.randn(batch_size, seq_length, d_model)
    encoder_block = EncoderBlock(d_model, num_heads, d_ff)

    output = encoder_block(x)

    assert output.shape == (batch_size, seq_length, d_model), "Output shape is incorrect!"
    print("✅ Encoder Block Test Passed!")

if __name__ == "__main__":
    test_encoder_block()
