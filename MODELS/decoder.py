#decoder block

import torch
import torch.nn as nn
from models.multi_head_attention import MultiHeadAttention
from models.feed_forward import FeedForward
from models.residual_layer_norm import ResidualConnection

class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.masked_attention = MultiHeadAttention(d_model, num_heads)
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        
        self.residual1 = ResidualConnection(d_model, dropout)
        self.residual2 = ResidualConnection(d_model, dropout)
        self.residual3 = ResidualConnection(d_model, dropout)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # Masked Self-Attention
        x = self.residual1(x, lambda x: self.masked_attention(x, x, x, tgt_mask))

        # Encoder-Decoder Attention
        x = self.residual2(x, lambda x: self.attention(x, encoder_output, encoder_output, src_mask))

        # Feed Forward Layer
        x = self.residual3(x, self.feed_forward)

        return x
