import torch
from models.Transformer import Transformer

input_vocab_size = 10000
target_vocab_size = 10000
d_model = 512
num_heads = 8
d_ff = 2048
num_layers = 6
dropout = 0.1

transformer = Transformer(input_vocab_size, target_vocab_size, d_model, num_heads, d_ff, num_layers, dropout)

batch_size = 2
seq_len = 10

src = torch.randint(0, input_vocab_size, (batch_size, seq_len), dtype=torch.long)
trg = torch.randint(0, target_vocab_size, (batch_size, seq_len), dtype=torch.long)

src_mask = torch.ones((batch_size, 1, 1, seq_len))  
trg_mask = torch.ones((batch_size, 1, seq_len, seq_len))

output = transformer(src, trg, src_mask, trg_mask)
print("Model output shape:", output.shape)
