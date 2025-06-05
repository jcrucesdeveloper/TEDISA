import torch

# Create tensors
x = torch.tensor([1, 2, 3, 4])
y = torch.tensor([5, 6, 7, 8])

# Basic operations
print("Addition:", x + y)
print("Multiplication:", x * y)
print("Mean:", x.mean())
print("Sum:", x.sum())

# Reshape and matrix operations
matrix = torch.tensor([[1, 2], [3, 4]])
print("\nMatrix:", matrix)
print("Transpose:", matrix.t())
print("Matrix multiplication:", torch.matmul(matrix, matrix))

# Random tensors
random_tensor = torch.rand(2, 3)
print("\nRandom tensor:", random_tensor)
print("Random tensor shape:", random_tensor.shape)

# GPU check (if available)
print("\nCUDA available:", torch.cuda.is_available())
