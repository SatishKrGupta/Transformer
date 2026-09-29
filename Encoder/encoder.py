import math
import torch
import torch.nn as nn

from Encoder.encoder_layer import EncoderLayer
from positional_encoding import PositionalEncoding


class Encoder(nn.Module):

    def __init__(
        self,
        vocab_size,
        d_model,
        num_heads,
        d_ff,
        num_layers,
        max_len=5000
    ):
        super().__init__()

        # 1. Token embedding
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model
        )

        # 2. Positional encoding
        self.positional_encoding = PositionalEncoding(
            d_model=d_model,
            max_len=max_len
        )

        # 3. Multiple encoder layers
        self.layers = nn.ModuleList([
            EncoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff
            )
            for _ in range(num_layers)
        ])

        self.d_model = d_model

    def forward(self, src):

        # src shape: [batch_size, seq_len]

        # Convert token IDs to embeddings
        x = self.embedding(src)

        # Scale embeddings
        x = x * math.sqrt(self.d_model)

        # Add positional information
        x = self.positional_encoding(x)

        # Pass through all encoder layers
        attention_weights = []

        for layer in self.layers:
            x, weights = layer(x)
            attention_weights.append(weights)

        return x, attention_weights




if __name__ == "__main__":

    vocab_size = 26
    d_model = 8
    num_heads = 2
    d_ff = 32
    num_layers = 2

    # Two sentences, five tokens each
    src = torch.tensor([
        [2, 8, 11, 9, 3],
        [2, 10, 12, 13, 3]
    ])

    encoder = Encoder(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_layers=num_layers
    )

    output, attention_weights = encoder(src)

    print("Input shape:", src.shape)
    print("Encoder output shape:", output.shape)
    print("Number of encoder layers:", len(attention_weights))
    print("Layer 1 attention shape:", attention_weights[0].shape)
    print("Layer 2 attention shape:", attention_weights[1].shape)