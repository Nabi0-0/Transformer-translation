import torch.nn as nn
from models.multi_head_attention import MultiHeadAttention
from models.feed_forward import FeedForward

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask, trg_mask):
        x = x + self.dropout(self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), trg_mask))
        x = x + self.dropout(self.cross_attn(self.norm2(x), self.norm2(enc_output), self.norm2(enc_output), src_mask))
        x = x + self.dropout(self.ffn(self.norm3(x)))
        return x
