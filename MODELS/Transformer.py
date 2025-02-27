#main transformer model

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import torch.nn as nn
from models.encoder import EncoderLayer
from models.decoder import DecoderLayer
from models.positional_encoding import PositionalEncoding
from models.embedding import WordEmbedding

class Transformer(nn.Module):
    def __init__(self, input_vocab_size, target_vocab_size, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super(Transformer, self).__init__()

        self.src_embedding = WordEmbedding(input_vocab_size, d_model)
        self.trg_embedding = WordEmbedding(target_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.fc_out = nn.Linear(d_model, target_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, trg, src_mask, trg_mask):
        src = self.dropout(self.pos_encoding(self.src_embedding(src)))
        trg = self.dropout(self.pos_encoding(self.trg_embedding(trg)))

        for layer in self.encoder_layers:
            src = layer(src, src_mask)

        for layer in self.decoder_layers:
            trg = layer(trg, src, src_mask, trg_mask)

        return self.fc_out(trg)
