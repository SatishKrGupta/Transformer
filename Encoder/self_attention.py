import torch
import torch.nn as nn
import math


class SelfAttention(nn.Module):

    def __init__(self, d_model, d_k):
        super().__init__()

        # Learnable projections
        self.W_Q = nn.Linear(d_model, d_k, bias=False)
        self.W_K = nn.Linear(d_model, d_k, bias=False)
        self.W_V = nn.Linear(d_model, d_k, bias=False)

    def forward(self, X):

        # 1. Create Query, Key and Value
        Q = self.W_Q(X)
        K = self.W_K(X)
        V = self.W_V(X)

        # 2. Calculate attention scores
        scores = torch.matmul(
            Q,
            K.transpose(-2, -1)
        )

        # 3. Scale the scores
        d_k = Q.size(-1)

        scores = scores / math.sqrt(d_k)

        # 4. Convert scores into attention weights
        attention_weights = torch.softmax(
            scores,
            dim=-1
        )

        # 5. Weighted sum of Values
        output = torch.matmul(
            attention_weights,
            V
        )

        return output, attention_weights


# ------------------------------------------------
# TEST
# ------------------------------------------------

if __name__ == "__main__":

    # Model dimension
    d_model = 4

    # Query / Key / Value dimension
    d_k = 4

    # One sentence containing 3 tokens
    X = torch.tensor([
        [0.2, 0.5, 0.1, 0.8],
        [0.7, 0.1, 0.4, 0.3],
        [0.9, 0.2, 0.6, 0.5]
    ])

    # Create self-attention
    self_attention = SelfAttention(
        d_model=d_model,
        d_k=d_k
    )

    # Run attention
    output, attention_weights = self_attention(X)

    print("Input X shape:")
    print(X.shape)

    print("\nQ shape:")
    print(self_attention.W_Q(X).shape)

    print("\nK shape:")
    print(self_attention.W_K(X).shape)

    print("\nV shape:")
    print(self_attention.W_V(X).shape)

    print("\nAttention weights:")
    print(attention_weights)

    print("\nOutput:")
    print(output)

    print("\nOutput shape:")
    print(output.shape)