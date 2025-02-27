#main transformer model
import torch
import torch.nn as nn
from models.encoder import EncoderBlock
from models.decoder import DecoderBlock

class Transformer(nn.Module):
    def __init__(self, input_vocab_size, target_vocab_size, d_model, num_heads, d_ff, num_layers, dropout):
        super(Transformer, self).__init__()

        self.encoder = EncoderBlock(d_model, num_heads, d_ff, dropout)        
        self.decoder = DecoderBlock(d_model, num_heads, d_ff, dropout)        
        self.fc_out = nn.Linear(d_model, target_vocab_size)  # Final output layer

    def forward(self, src, trg, src_mask, trg_mask):
        enc_output = self.encoder(src, src_mask)
        dec_output = self.decoder(trg, enc_output, src_mask, trg_mask)
        output = self.fc_out(dec_output)
        return output
