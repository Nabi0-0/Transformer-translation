#for testing postional encoding


import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from positional_encoding import PositionalEncoding

#define model size and sequence length
d_model = 512
seq_len = 10

#initializa positional encoding
pos_enc = PositionalEncoding(d_model)

#create a dummy input tensor of shape (batch size = 1, seq_len = 10, d_model = 512)
x = torch.zeros(1, seq_len,d_model)

#pass it through positional encoding
encoded_x = pos_enc(x)

#print output spape
print("output shape: ", encoded_x.shape) #should be (1, 10, 512)