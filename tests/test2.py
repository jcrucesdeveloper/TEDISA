import torch

# Create a 3D tensor
tensor_3d = torch.tensor([
    [[1, 2], [3, 4]],
    [[5, 6], [7, 8]],
    [[9, 10], [11, 12]]
])

print("Original 3D tensor:", tensor_3d)
print("Shape:", tensor_3d.shape)

# Indexing operations
print("\nFirst element:", tensor_3d[0])
print("First row of first element:", tensor_3d[0, 0])
print("Specific value:", tensor_3d[1, 1, 1])

# Slicing operations
print("\nFirst two elements:", tensor_3d[:2])
print("Last row of each element:", tensor_3d[:, -1])
print("Alternate elements:", tensor_3d[::2])

# Advanced indexing
indices = torch.tensor([0, 2])
print("\nSelected elements:", tensor_3d[indices])

# Boolean indexing
mask = tensor_3d > 5
print("\nElements > 5:", tensor_3d[mask]) 