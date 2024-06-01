import torch
import torch.nn as nn
import math
import torch.nn.functional as F


def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size(-1)
    scaled = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        # Expand the mask to match the dimensions of scaled
        mask = mask.unsqueeze(1).unsqueeze(1)  # Add two dimensions to match the shape of scaled

        # Check if dimensions match before adding
        assert scaled.size()[-2:] == mask.size()[-2:], "Mask dimensions do not match with scaled dimensions"

        scaled += mask * (-1e9)  # adding -1e9 to the positions where mask is 1 to make them very negative

    attention = F.softmax(scaled, dim=-1)
    values = torch.matmul(attention, v)

    return values, attention


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv_layer = nn.Linear(d_model, 3 * d_model)
        self.linear_layer = nn.Linear(d_model, d_model)

    def forward(self, x, mask):
        batch_size, sequence_length, d_model = x.size()

        qkv = self.qkv_layer(x)
        qkv = qkv.reshape(batch_size, sequence_length, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        values, attention = scaled_dot_product(q, k, v, mask)

        values = values.permute(1, 0, 2, 3).reshape(batch_size, sequence_length, self.num_heads * self.head_dim)

        out = self.linear_layer(values)

        return out
