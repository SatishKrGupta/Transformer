import torch

# Check PyTorch version
print("PyTorch version:", torch.__version__)

# Check whether GPU is available
print("GPU available:", torch.cuda.is_available())

# Create a tensor
x = torch.tensor([1, 2, 3, 4, 5])

print("Our first tensor:", x)