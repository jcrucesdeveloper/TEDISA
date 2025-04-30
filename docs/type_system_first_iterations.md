# Type System first iteration

We need to capture technical requirements for implementing a type system that first captures a Tensor's dimensions and potentially its shape. This will use simple functions from the [PyTorch documentation](https://pytorch.org/docs/stable/index.html). The goal is capture the tensor's shape and dimensions in a type.

### First Approach to Typing Tensor Dimensions and Shapes

The following examples use PyTorch methods that create or modify tensor dimensions and shapes. This represents an initial approach to developing a type system for tracking these properties.

### Example 1: torch.tensor

Takes a Python array as input and returns a tensor with the dimensions and shape of that array. The dimensions represent the number of axes (rank) while the shape indicates the size along each dimension.

**Parameters:**

- `data` (`array_like`): Initial data for the tensor. Can be a list, tuple, NumPy ndarray, scalar, and other types.

```python

;; tensor :: List[Number] -> Tensor ([x_1, x_2, ..., x_n], dim=n)
x = torch.tensor([1, 2, 3])
x.dim() # Returns 1
x.shape # Returns torch.Size([3])
```

### Example 2: torch.flatten

Flattens input by reshaping it into a one-dimensional tensor. The order of elements in input is unchanged.

**Parameters:**

- `input` (`Tensor`): The input tensor.

```python
;; flatten :: Tensor ([x_1, x_2, ... , x_n], dim=n) -> Tensor ([x_1 * x_2 * ... * x_n], dim=1)
t = torch.tensor([[[1, 2],
                   [3, 4]],
                  [[5, 6],
                   [7, 8]]]) # Shape (2,2,2)

torch.flatten(t)
>>> tensor([1,2,3,4,5,6,7,8,9]) # shape (8)
torch.flatten(t, start_dim=1)
```

### Example 3: torch.reshape

Returns a tensor with the same data and number of elements as input, but with the specified shape.
**Parameters:**

- `input` (`Tensor`): The tensor to be reshaped
- `shape` (`tuple of int`): The new shape

```python
;; reshape :: Tensor ([x_1, x_2, ... , x_n], dim=n) Tuple(y_1, y_2, ..., y_m) -> Tensor ([y_1, y_2, ..., y_m], dim=m)
t = torch.tensor([[[1, 2],
                   [3, 4]],
                  [[5, 6],
                   [7, 8]]]) # Shape (2,2,2)

a = torch.zeros(4) # [0, 0, 0, 0], shape(4) dim=1
torch.reshape(a, (2, 2))
>>> tensor([[ 0.,  0.],
>>>         [ 0.,  0.]]) # Shape (2,2) dim=2

```

### Example 4: torch.permute

Returns a view of the original tensor input with its dimensions permuted.

**Parameters:**

- `input` (`Tensor`): The input tensor.
- `dims` (`tuple of int`): The desired ordering of dimensions

```python
;; permute :: Tensor([x_1, x_2, ..., x_n], dim=n) Tuple(y_1, y_2, ... , y_n) -> Tensor([Tensor[y_1].shape, Tensor[y_2].shape, ... , Tensor[y_n].shape], dim=n)
x = torch.rand(2,3,5)
x.shape()
>>> torch.Size([2,3,5])
torch.permute(x, (2, 0, 1)).shape()
>>> torch.size([5,2,3])
```

### Example 5: torch.cat

Concatenates the given sequence of tensors in tensors in the given dimension. All tensors must either have the same shape (except in the concatenating dimension) or be a 1-D empty tensor with size (0,).

**Parameters:**

- `tensors` (`sequence of Tensors`): Non-empty tensors provided must have the same shape, except in the cat dimension
- `dim` (`int`, optional): The dimension along which the tensors are concatenated

```python
;; cat :: List[Tensor([x_1, ..., x_n], dim=n), Tensor([y_1, ...,y_n], dim=n), ... , Tensor[n_1, n_]] Int=m -> Tensor([x_1, x_2, ..., x_m + y_m, ..., x_n], dim=n)
;; m is the dimension where we are concadenating
x = torch.randn(2, 3)  # Shape (2,3)
y = torch.randn(2, 3)  # Shape (2,3)
torch.cat((x, y), dim=0)
>>> torch.Size([4, 3])
```

### Example 6: torch.squeeze

Removes dimensions of size 1 from the tensor.

**Parameters:**

- `input` (`Tensor`): The input tensor.
- `dim` (`int`, optional): If given, the input will be squeezed only in this dimension.

```python
;; squeeze :: Tensor([x_1, x_2, ..., 1, ..., x_n], dim=n) -> Tensor([x_1, x_2, ..., x_m], dim=m) where m <= n
x = torch.zeros(2, 1, 2, 1, 2)  # Shape (2,1,2,1,2)
x.shape
>>> torch.Size([2, 1, 2, 1, 2])
torch.squeeze(x).shape
>>> torch.Size([2, 2, 2])  # Removed all dimensions of size 1

# With dim parameter
y = torch.zeros(2, 1, 2, 1, 2)
torch.squeeze(y, dim=1).shape
>>> torch.Size([2, 2, 1, 2])  # Only removed dimension at index 1
```
