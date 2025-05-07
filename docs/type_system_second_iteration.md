# Type System first iteration

We need to capture technical requirements for implementing a type system that first captures a Tensor's dimensions and potentially its shape. This will use simple functions from the [PyTorch documentation](https://pytorch.org/docs/stable/index.html). The goal is capture the tensor's shape and dimensions in a type.

### First Approach to Typing Tensor Dimensions and Shapes

The following examples use PyTorch methods that create or modify tensor dimensions and shapes. This represents an initial approach to developing a type system for tracking these properties.

### Example 1: torch.tensor

Takes a Python array as input and returns a tensor with the dimensions and shape of that array. The dimensions represent the number of axes (rank) while the shape indicates the size along each dimension.

**Parameters:**

- `data` (`array_like`): Initial data for the tensor. Can be a list, tuple, NumPy ndarray, scalar, and other types.

```python
;; tensor :: List^k[Number] (shape=[d1, d2, ..., dk]) -> Tensor([d1, d2, ..., dk], dim=k)

; k = number of nested list levels
; [d1, d2, ..., dk] = lengths at each level of nesting

# Dimension and Shape - dim 1
d1 = torch.tensor([1, 2, 3])
d1.dim() # 1
d1.size() # torch.Size([3])

# Dimension and Shape - dim 2
d2 = torch.tensor([[1, 2, 3], [4, 5, 6]])
d2.dim() # 2
d2.size() # torch.Size([2, 3])

# Dimension and Shape - dim 3
d3 = torch.tensor([
    [[1, 2], [3, 4]],
    [[5, 6], [7, 8]]
])
d3.dim() # 3
d3.size() # torch.Size([2, 2, 2])
```

### Example 2: torch.flatten

Flattens input by reshaping it into a one-dimensional tensor. The order of elements in input is unchanged.

**Parameters:**

- `input` (`Tensor`): The input tensor.

```python
;; flatten :: Tensor ([x_1, x_2, ... , x_n], dim=n) -> Tensor ([x_1 * x_2 * ... * x_n], dim=1)

# Dimension and Shape - before flatten
t = torch.tensor([[[1, 2],
                   [3, 4]],
                  [[5, 6],
                   [7, 8]]])
t.dim() # 3
t.size() # torch.Size([2, 2, 2])

# Dimension and Shape - after flatten
t = torch.flatten(t)
t.dim() # 1
t.size() # torch.Size([8])

print(t)
>>> tensor([1,2,3,4,5,6,7,8])
```

### Example 3: torch.reshape

Returns a tensor with the same data and number of elements as input, but with the specified shape.

**Parameters:**

- `input` (`Tensor`): The tensor to be reshaped
- `shape` (`tuple of int`): The new shape

**Constraints**

- All other dimensions must be positive integers.

```python
# Runtime Error - Negative dimension
t = torch.zeros(4)
torch.reshape(t, (-2,2))
>>> RuntimeError: invalid shape dimension -2
```

- Only one dimension can be -1 (PyTorch will infer its value).

```python
# Runtime Error - Multiple -1 dimensions
t = torch.zeros(4)
torch.reshape(t, (-1,-1))
>>> RuntimeError: only one dimension can be inferred
```

- The product of the new shape's dimensions should be equal to the product of the original shape's dimensions

```python
# Runtime Error - Product of shapes not equals
t = torch.zeros(4, 2)
torch.reshape(t, (4, 3))
# The product of new shape dimensions (4*3=12) must equal to (4*2=8)
>>> RuntimeError: shape '[4, 3]' is invalid for input of size 8
```

```python
;; reshape :: Tensor ([x_1, x_2, ... , x_n], dim=n) Tuple(y_1, y_2, ..., y_m) -> Tensor ([y_1, y_2, ..., y_m], dim=m)
; constraints:
; ∀i ∈ [1..m]: y_i > 0 ∨ y_i = -1
; |{i | y_i = -1}| ≤ 1
; ∏(y_i) = ∏(x_i)

# Dimension and Shape - before reshape
t = torch.zeros(4) # [0, 0, 0, 0], shape(4) dim=1
t.dim() # 1
t.size() # torch.Size([4])

# Dimension and Shape - after reshape
t = torch.reshape(a, (2, 2))
t.dim() # 2
t.size() # torch.Size([2, 2])

print(b)
>>> tensor([[ 0.,  0.],
>>>         [ 0.,  0.]]) # Shape (2,2) dim=2
```

### Example 4: torch.permute

Returns a view of the original tensor input with its dimensions permuted.

**Parameters:**

- `input` (`Tensor`): The input tensor.
- `dims` (`tuple of int`): The desired ordering of dimensions

**Constraints:**

- Dimensions must be in range [-n, n - 1] where n is the number of dimensions

```python
# IndexError- Dimension out of range
t = torch.rand(2,3,5)  # 3 dimensions
torch.permute(t, (3, 0, 1))  # 3 is out of range [-3, 2]
>>> IndexError: Dimension out of range (expected to be in range of [-3, 2], but got 3)
```

- No dimension can be repeated

```python
# RuntimeError - Repeated dimension
t = torch.rand(2,3,5)
torch.permute(t, (0, 0, 1))  # 0 is repeated
>>> RuntimeError: permute(): duplicate dims are not allowed.
```

- The number of dimensions in the new shape should be equal to the original number of dimensions

```python
# Runtime Error - Invalid shape
t = torch.rand(2,3,5)
torch.permute(t, (0, 1))
>>> RuntimeError: permute(sparse_coo): number of dimensions in the tensor input does not match the length of the desired ordering of dimensions i.e. input.dim() = 3 is not equal to len(dims) = 2
```

```python
;; permute :: Tensor([x_1, x_2, ..., x_n], dim=n) Tuple(y_1, y_2, ... , y_m) -> Tensor([Tensor[y_1].shape, Tensor[y_2].shape, ... , Tensor[y_n].shape], dim=n)
; constraints:
; ∀i ∈ [1..m]: -n ≤ y_i < n - 1
; ∀i,j ∈ [1..m]: i != j → y_i != y_j
; m = n

# Dimension and Shape - before permute
t = torch.rand(2,3,5)
t.dim() # 3
t.size() # torch.Size([2,3,5])

# Dimension and Shape - after permute
t = torch.permute(t, (2, 0, 1))
t.dim() # 3
t.size() # torch.Size([5,2,3])
```

### Example 5: torch.cat

Concatenates the given sequence of tensors in tensors in the given dimension. All tensors must either have the same shape (except in the concatenating dimension) or be a 1-D empty tensor with size (0,).

**Parameters:**

- `tensors` (`sequence of Tensors`): Non-empty tensors provided must have the same shape, except in the cat dimension
- `dim` (`int`, optional): The dimension along which the tensors are concatenated

**Constraints:**

- Dimensions must be in range [-n, n - 1] where n is the number of dimensions

```python
# IndexError - Dimension out of range
x = torch.randn(2, 3)  # Shape (2,3)
y = torch.randn(2, 3)  # Shape (2,3)
torch.cat((x, y), dim=2)  # dim 2 is out of range [-2, 1]
>>> IndexError: Dimension out of range (expected to be in range of [-2, 1], but got 2)
```

- All tensors must have the same shape except in the concatenating dimension

```python
# RuntimeError - Shapes don't match
x = torch.randn(2, 3)  # Shape (2,3)
y = torch.randn(2, 4)  # Shape (2,4)
torch.cat((x, y), dim=0)
>>> RuntimeError: Sizes of tensors must match except in dimension 0. Expected size 3 but got size 4 for tensor number 1 in the list.
```

```python
;; cat :: (Tensor([x_1, ..., x_n], dim=n), ... , Tensor([y_1, ...,y_n], dim=n))  Int=m -> Tensor([x_1, x_2, ..., x_m + y_m, ..., x_n], dim=n)
; constraints:
; -n <= m < n
; ∀t1,t2 ∈ tensors: ∀i ∈ [1..n]: i ≠ m → t1.shape[i] = t2.shape[i]

# Dimension and Shape - after cat
x = torch.randn(2, 3)  # Shape (2,3)
y = torch.randn(2, 3)  # Shape (2,3)
t = torch.cat((x, y), dim=0)
t.dim() # 2
t.size() # torch.Size([4, 3])
```
