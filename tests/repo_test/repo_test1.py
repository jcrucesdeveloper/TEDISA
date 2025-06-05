import torch

# Create matrices for linear algebra operations
A = torch.tensor([[1., 2.], [3., 4.]])
B = torch.tensor([[5., 6.], [7., 8.]])

# Basic linear algebra operations
print("Matrix A:", A)
print("Matrix B:", B)

# Matrix decomposition
print("\nLU decomposition:")
lu, pivots = torch.lu(A)
print("LU:", lu)
print("Pivots:", pivots)

# QR decomposition
print("\nQR decomposition:")
Q, R = torch.qr(A)
print("Q:", Q)
print("R:", R)

# SVD decomposition
print("\nSVD decomposition:")
U, S, V = torch.svd(A)
print("U:", U)
print("S:", S)
print("V:", V)

# Matrix properties
print("\nMatrix properties:")
print("Determinant:", torch.det(A))
print("Matrix rank:", torch.matrix_rank(A))
print("Eigenvalues:", torch.eigvals(A))

# Linear system solving
b = torch.tensor([1., 2.])
print("\nSolving Ax = b:")
x = torch.linalg.solve(A, b)
print("Solution x:", x)

# Create tensors
tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
print("\nOriginal tensor:", tensor)

# Statistical operations
print("\nVariance:", tensor.var())
print("Standard deviation:", tensor.std())
print("Median:", tensor.median())
print("Mode:", torch.mode(tensor))
print("Quantile (0.25):", torch.quantile(tensor, 0.25))

# Reduction operations
tensor_2d = torch.tensor([[1, 2, 3], [4, 5, 6]])
print("\n2D tensor:", tensor_2d)
print("Max along dim 0:", tensor_2d.max(dim=0))
print("Min along dim 1:", tensor_2d.min(dim=1))
print("Argmax:", tensor_2d.argmax())
print("Argmin:", tensor_2d.argmin())

# Advanced statistics
print("\nCumulative sum:", tensor.cumsum(0))
print("Cumulative product:", tensor.cumprod(0))
print("Unique values:", torch.unique(tensor))
print("Histogram:", torch.histc(tensor, bins=3)) 