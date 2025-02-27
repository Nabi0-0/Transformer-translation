#for testing decoder block

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from models.decoder import DecoderBlock

def test_decoder_block():
    d_model = 512
    num_heads = 8
    d_ff = 2048
    dropout = 0.1
    batch_size = 2
    seq_len = 10

    decoder_block = DecoderBlock(d_model, num_heads, d_ff, dropout)

    x = torch.randn(batch_size, seq_len, d_model)
    encoder_output = torch.randn(batch_size, seq_len, d_model)
    src_mask = torch.ones(batch_size, 1, 1, seq_len)
    tgt_mask = torch.ones(batch_size, 1, seq_len, seq_len)

    output = decoder_block(x, encoder_output, src_mask, tgt_mask)
    print("Decoder output shape:", output.shape)

test_decoder_block()
