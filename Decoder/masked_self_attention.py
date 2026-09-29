import torch
import torch.nn as nn
import math


class MaskedSelfAttention(nn.Module):

    def __init__(self, d_model, num_heads):
        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)

        self.W_O = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):

        batch_size, seq_len, _ = x.shape

        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        # Split into attention heads
        Q = Q.view(
            batch_size, seq_len,
            self.num_heads, self.d_head
        ).transpose(1, 2)

        K = K.view(
            batch_size, seq_len,
            self.num_heads, self.d_head
        ).transpose(1, 2)

        V = V.view(
            batch_size, seq_len,
            self.num_heads, self.d_head
        ).transpose(1, 2)

        # Calculate attention scores
        scores = torch.matmul(
            Q, K.transpose(-2, -1)
        ) / math.sqrt(self.d_head)

        # Lower-triangular causal mask
        mask = torch.tril(
            torch.ones(seq_len, seq_len, device=x.device)
        ).bool()

        # Block future positions
        scores = scores.masked_fill(
            ~mask, float("-inf")
        )

        attention_weights = torch.softmax(
            scores, dim=-1
        )

        output = torch.matmul(
            attention_weights, V
        )

        # Join attention heads
        output = output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )

        output = self.W_O(output)

        return output, attention_weights


if __name__ == "__main__":

    x = torch.randn(2, 4, 8)

    model = MaskedSelfAttention(
        d_model=8,
        num_heads=2
    )

    output, weights = model(x)

    print("Input:", x.shape)
    print("Output:", output.shape)
    print("Attention weights:", weights.shape)

    print("\nHead 1, first batch:")
    print(weights[0, 0])

    # Verify future positions receive zero attention
    future_weights = torch.triu(
        weights[0, 0], diagonal=1
    )

    print("\nFuture attention weights:")
    print(future_weights)

    assert torch.all(future_weights == 0)
    print("\nCausal mask test passed!")