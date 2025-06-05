import torch

# Create initial tensor
tensor = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8])
print("Original tensor:", tensor)

# Reshape operations
reshaped = tensor.reshape(2, 4)
print("\nReshaped to 2x4:", reshaped)

# View operations
viewed = tensor.view(4, 2)
print("Viewed as 4x2:", viewed)

# Squeeze and unsqueeze
tensor_3d = torch.tensor([[[1], [2]], [[3], [4]]])
print("\nOriginal 3D tensor:", tensor_3d)
print("Squeezed:", tensor_3d.squeeze())
print("Unsqueezed at dim 0:", tensor.unsqueeze(0))

# Permute dimensions
tensor_2d = torch.tensor([[1, 2], [3, 4], [5, 6]])
print("\nOriginal 2D tensor:", tensor_2d)
print("Permuted dimensions:", tensor_2d.permute(1, 0))

# Flatten and unflatten
print("\nFlattened:", tensor_2d.flatten())
print("Unflattened:", tensor_2d.flatten().unflatten(0, (2, 3))) 