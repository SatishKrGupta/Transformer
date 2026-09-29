
import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        # Matrix containing positional encodings
        pe = torch.zeros(max_len, d_model)

        # Positions: 0, 1, 2, ..., max_len - 1
        position = torch.arange(max_len).unsqueeze(1)

        # Frequency values for each pair of dimensions
        div_term = torch.exp(
            torch.arange(0, d_model, 2)
            * (-math.log(10000.0) / d_model)
        )

        # Sine for even dimensions
        pe[:, 0::2] = torch.sin(position * div_term)

        # Cosine for odd dimensions
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension: (1, max_len, d_model)
        pe = pe.unsqueeze(0)

        # Store as a buffer, not a trainable parameter
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, d_model)
        seq_len = x.size(1)

        # Add positional information to token embeddings
        return x + self.pe[:, :seq_len, :]


# Test our positional encoding
if __name__ == "__main__":
    d_model = 4
    seq_len = 7
    batch_size = 1

    x = torch.zeros(batch_size, seq_len, d_model)

    positional_encoding = PositionalEncoding(d_model)

    output = positional_encoding(x)

    print("Positional encoding shape:", output.shape)
    print("\nPositional encodings:")
    print(output[0])