import torch
import torch.nn as nn

from Encoder.multi_head_attention import MultiHeadAttention


class FeedForward(nn.Module):

    def __init__(self, d_model, d_ff):
        super().__init__()

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.linear1(x)

        x = self.relu(x)

        x = self.linear2(x)

        return x


class EncoderLayer(nn.Module):

    def __init__(self, d_model, num_heads, d_ff):

        super().__init__()

        self.self_attention = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads
        )

        self.feed_forward = FeedForward(
            d_model=d_model,
            d_ff=d_ff
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):

        # 1. Multi-Head Self-Attention
        attention_output, attention_weights = self.self_attention(x)

        # 2. Residual Connection + LayerNorm
        x = self.norm1(x + attention_output)

        # 3. Feed-Forward Network
        ff_output = self.feed_forward(x)

        # 4. Residual Connection + LayerNorm
        x = self.norm2(x + ff_output)

        return x, attention_weights