# Important PyTorch Definitions

This documentation is extracted from the [PyTorch documentation](https://pytorch.org/docs/stable/).

## torch.Tensor.size

Tensor.size(dim=None) → torch.Size or int

Returns the size of the self tensor. If dim is not specified, the returned value is a torch.Size, a subclass of tuple. If dim is specified, returns an int holding the size of that dimension.

### Parameters

dim (int, optional) – The dimension for which to retrieve the size.

#### Examples

```python
>>> t = torch.empty(3, 4, 5)
>>> t.size()
torch.Size([3, 4, 5])
>>> t.size(dim=1)
4
```

### torch.Size

torch.Size is the result type of a call to torch.Tensor.size(). It describes the size of all dimensions of the original tensor. As a subclass of tuple, it supports common sequence operations like indexing and length.

#### Examples

```python
>>> x = torch.ones(10, 20, 30)
>>> s = x.size()
>>> s
torch.Size([10, 20, 30])
>>> s[1]
20
>>> len(s)
3
```

### So, what does the size/shape of a Tensor mean?

It describes how many elements are in each dimension of the tensor:

- If you ask for a specific dimension, it returns an integer with the size of that dimension.
- If you ask for the size of the whole tensor, it returns a tuple with the size of every dimension.

## torch.Tensor.dim

Tensor.dim() → int

Returns the number of dimensions of self tensor.

#### Examples

```python
>>> # 0D tensor (scalar)
>>> x = torch.tensor(5)
>>> x.dim()
0

>>> # 1D tensor (vector)
>>> x = torch.tensor([1, 2, 3])
>>> x.dim()
1

>>> # 2D tensor (matrix)
>>> x = torch.tensor([[1, 2], [3, 4]])
>>> x.dim()
2

>>> # 3D tensor
>>> x = torch.ones(2, 3, 4)
>>> x.dim()
3
```

### Differences between shape/size and dim of a Tensor

- Shape/size: The number of elements in each dimension
- Dim: The number of dimensions
