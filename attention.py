import torch
import math


def scaled_dot_product_attention(Q, K, V):

    # Step 1: Calculate QK^T
    scores = torch.matmul(Q, K.transpose(-2, -1))

    # Step 2: Scale by sqrt(d_k)
    d_k = Q.size(-1)
    scores = scores / math.sqrt(d_k)

    # Step 3: Convert scores into probabilities
    attention_weights = torch.softmax(scores, dim=-1)

    # Step 4: Weighted sum of Values
    output = torch.matmul(attention_weights, V)

    return output, attention_weights


# -------------------------
# Test
# -------------------------

Q = torch.tensor([
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0]
])

K = torch.tensor([
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0]
])

V = torch.tensor([
    [10.0, 20.0],
    [20.0, 30.0],
    [30.0, 40.0]
])


output, attention_weights = scaled_dot_product_attention(Q, K, V)

print("Attention weights:")
print(attention_weights)

print("\nAttention output:")
print(output)