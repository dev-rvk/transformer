import torch
import torch.nn as nn
import math


class InputEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(InputEmbedding, self).__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, tokens):
        return self.embedding(tokens) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)
        # Compute positional encodings once in log space
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # print(x.shape)
        # print(self.pe.shape)
        pe = self.pe[:, :x.shape[0], :].squeeze(0)
        # print(pe.shape)
        x = x + pe
        return self.dropout(x)


class AbsolutePositionalEncoder(nn.Module):
    def __init__(self, emb_dim, max_position=512):
        super(AbsolutePositionalEncoder, self).__init__()
        self.position = torch.arange(max_position).unsqueeze(1)

        self.positional_encoding = torch.zeros(1, max_position, emb_dim)

        _2i = torch.arange(0, emb_dim, step=2).float()

        # PE(pos, 2i) = sin(pos/10000^(2i/d_model))
        self.positional_encoding[0, :, 0::2] = torch.sin(self.position / (10000 ** (_2i / emb_dim)))

        # PE(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        self.positional_encoding[0, :, 1::2] = torch.cos(self.position / (10000 ** (_2i / emb_dim)))

    def forward(self, x):
        # batch_size, input_len, embedding_dim
        batch_size, seq_len = x.size()
        return self.positional_encoding[:batch_size, :seq_len, :]


# # Example usage:
# # Initialize TransformerEmbedding instance
# vocab_size = 100256  # Example vocabulary size
# embedding_dim = 512  # Example embedding dimensionality
#
# input_embedding = InputEmbedding(vocab_size, embedding_dim)
# positional_encoding = PositionalEncoding(embedding_dim)
#
# # Example tokens
# tokens = torch.tensor([15339, 0, 32])  # Example tokens
#
# # Get embeddings for tokens
# input_embeddings = input_embedding(tokens)
# embeddings = positional_encoding(input_embeddings)
#
# print("Embedding", input_embeddings.shape)
#
# print("Embeddings shape:", embeddings.shape)

