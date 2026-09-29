
import torch
import torch.nn as nn

# Our tiny example vocabulary
vocab_size = 26

# Number of values representing each token
embedding_dim = 4

# Create the embedding layer
embedding = nn.Embedding(
    num_embeddings=vocab_size,
    embedding_dim=embedding_dim
)

# Token IDs from our Hindi tokenizer
token_ids = torch.tensor([2, 18, 17, 22, 16, 25, 3])

# Convert token IDs into embedding vectors
vectors = embedding(token_ids)

print("Token IDs:")
print(token_ids)

print("\nEmbedding vectors:")
print(vectors)

print("\nEmbedding shape:")
print(vectors.shape)