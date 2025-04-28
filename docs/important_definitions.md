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

### So, what does the size/shape of a Tensor means?

A representation of the length of each dimension of the original vector:

- If the vector has dimension one, it return an int with the length of that vector
- If the dimension of the vector are greater than one, it returns a tuple with all teh dimensions
