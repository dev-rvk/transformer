import torch
import torch.nn as nn
from tokenizer.tokenizer import Tokenizer
from embedding.embedding import InputEmbedding, PositionalEncoding
from MultiHeadAttention.multiheadattention import MultiHeadAttention
from normalization.normalization import LayerNormalization
from feedfordward.feedforward import FeedForward


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()

        self.multihead_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)

        self.layer_norm1 = LayerNormalization(d_model)
        self.layer_norm2 = LayerNormalization(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        # Multi-Head Attention
        if mask is not None:
            attn_out, _ = self.multihead_attention(x, mask)
        else:
            attn_out = self.multihead_attention(x, mask)
        attn_out = self.dropout1(attn_out)
        x = self.layer_norm1(x + attn_out)

        # Feed-Forward Network
        ff_out = self.feed_forward(x)
        ff_out = self.dropout2(ff_out)
        x = self.layer_norm2(x + ff_out)

        return x


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_length=5000, dropout=0.1):
        super(Encoder, self).__init__()

        self.embedding = InputEmbedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])

    def forward(self, x, mask):
        x = self.embedding(x)
        x = self.positional_encoding(x)

        for layer in self.layers:
            x = layer(x, mask)

        return x


# VOCAB_SIZE = 100256
# D_MODEL = 512
# NUM_HEADS = 8
# D_FF = 2048
#
# encoder = Encoder(d_model=D_MODEL,
#                   vocab_size=VOCAB_SIZE,
#                   num_heads=NUM_HEADS,
#                   d_ff=D_FF,
#                   max_seq_length=5000,
#                   num_layers=5)
#
# tokens = torch.tensor([15339, 0, 32, 0])  # Example tokens
#
#
# def create_padding_mask(seq):
#     mask = (seq == 0)
#     return mask.unsqueeze(1).unsqueeze(2)
#
#
# # Create padding mask for the input
# padding_mask = None
#
# encoded_out = encoder(tokens.unsqueeze(0), padding_mask)
#
# print(encoded_out.shape)
# print(encoded_out)
