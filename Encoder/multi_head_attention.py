import torch
import torch.nn as nn
import math


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, num_heads):
        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads

        # Dimension handled by each head
        self.d_head = d_model // num_heads

        # Q, K and V projections
        self.W_Q = nn.Linear(
            d_model,
            d_model,
            bias=False
        )

        self.W_K = nn.Linear(
            d_model,
            d_model,
            bias=False
        )

        self.W_V = nn.Linear(
            d_model,
            d_model,
            bias=False
        )

        # Final projection
        self.W_O = nn.Linear(
            d_model,
            d_model,
            bias=False
        )

    def forward(self, X):

        batch_size, seq_len, _ = X.shape

        # --------------------------------
        # 1. Create Q, K, V
        # --------------------------------

        Q = self.W_Q(X)
        K = self.W_K(X)
        V = self.W_V(X)

        # --------------------------------
        # 2. Split into multiple heads
        # --------------------------------

        Q = Q.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.d_head
        )

        K = K.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.d_head
        )

        V = V.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.d_head
        )

        # Move heads before sequence dimension

        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Shape:
        # [batch, heads, seq_len, d_head]

        # --------------------------------
        # 3. Attention scores
        # --------------------------------

        scores = torch.matmul(
            Q,
            K.transpose(-2, -1)
        )

        # --------------------------------
        # 4. Scale
        # --------------------------------

        scores = scores / math.sqrt(self.d_head)

        # --------------------------------
        # 5. Softmax
        # --------------------------------

        attention_weights = torch.softmax(
            scores,
            dim=-1
        )

        # --------------------------------
        # 6. Weighted sum of V
        # --------------------------------

        attention_output = torch.matmul(
            attention_weights,
            V
        )

        # --------------------------------
        # 7. Concatenate heads
        # --------------------------------

        attention_output = attention_output.transpose(1, 2)

        attention_output = attention_output.contiguous().view(
            batch_size,
            seq_len,
            self.d_model
        )

        # --------------------------------
        # 8. Final projection
        # --------------------------------

        output = self.W_O(attention_output)

        return output, attention_weights


# =========================================
# TEST
# =========================================

if __name__ == "__main__":

    batch_size = 2
    seq_len = 5
    d_model = 8
    num_heads = 2

    # Example input
    X = torch.randn(
        batch_size,
        seq_len,
        d_model
    )

    model = MultiHeadAttention(
        d_model=d_model,
        num_heads=num_heads
    )

    output, weights = model(X)

    print("Input shape:")
    print(X.shape)

    print("\nOutput shape:")
    print(output.shape)

    print("\nAttention weights shape:")
    print(weights.shape)