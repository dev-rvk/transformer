import torch
import torch.nn as nn
from tokenizer.tokenizer import Tokenizer
from embedding.embedding import InputEmbedding, PositionalEncoding
from MultiHeadAttention.multiheadattention import MultiHeadAttention
from normalization.normalization import LayerNormalization
from feedfordward.feedforward import FeedForward


class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderBlock, self).__init__()

        # Masked Multi-Head Self-Attention
        self.masked_multihead_attention = MultiHeadAttention(d_model, num_heads)
        self.layer_norm1 = LayerNormalization(d_model)
        self.dropout1 = nn.Dropout(dropout)

        # Multi-Head Attention with Encoder Output
        self.encoder_attention = MultiHeadAttention(d_model, num_heads)
        self.layer_norm2 = LayerNormalization(d_model)
        self.dropout2 = nn.Dropout(dropout)

        # Position-wise Feed-Forward Network
        self.feed_forward = FeedForward(d_model, d_ff)
        self.layer_norm3 = LayerNormalization(d_model)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, encoder_output, self_mask, encoder_mask):
        # Masked Multi-Head Self-Attention
        masked_attention_out, _ = self.masked_multihead_attention(x, self_mask)
        masked_attention_out = self.dropout1(masked_attention_out)
        x = self.layer_norm1(x + masked_attention_out)

        # Multi-Head Attention with Encoder Output
        encoder_attention_out, _ = self.encoder_attention(x, encoder_output, encoder_mask)
        encoder_attention_out = self.dropout2(encoder_attention_out)
        x = self.layer_norm2(x + encoder_attention_out)

        # Position-wise Feed-Forward Network
        ff_out = self.feed_forward(x)
        ff_out = self.dropout3(ff_out)
        x = self.layer_norm3(x + ff_out)

        return x


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_length=5000, dropout=0.1):
        super(Decoder, self).__init__()

        self.embedding = InputEmbedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.layers = nn.ModuleList([
            DecoderBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])

        self.output_layer = nn.Linear(d_model, vocab_size)

    def forward(self, x, encoder_output, self_mask, encoder_mask):
        x = self.embedding(x)
        x = self.positional_encoding(x)

        for layer in self.layers:
            x = layer(x, encoder_output, self_mask, encoder_mask)

        output = self.output_layer(x)

        return output
