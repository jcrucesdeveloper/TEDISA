# Type System second approach iteration

To capture technical requirements for implementing a type system that first captures a Tensor's dimensions and potentially its shape. This will use simple functions from the [PyTorch documentation](https://pytorch.org/docs/stable/index.html). The goal is capture the tensor's shape and dimensions in a type.

## First Approach to Typing Tensor Dimensions and Shapes

The following examples use PyTorch methods that create or modify tensor dimensions and shapes. This represents an initial approach to developing a type system for tracking these properties.

## Type System Approach

The type system tracks tensor dimensions and shapes through a formal notation:

- `Tensor([d₁, d₂, ..., dₙ], dim=n)`: Represents a tensor with shape [d₁, d₂, ..., dₙ] and n dimensions
- Functions are typed with input/output types and constraints
- Constraints are expressed using first-order logic predicates

Example:

```python
;; reshape :: Tensor([x₁, x₂, ..., xₙ], dim=n) Tuple(y₁, y₂, ..., yₘ) -> Tensor([y₁, y₂, ..., yₘ], dim=m)
; constraints:
; ∀i ∈ [1..m]: yᵢ > 0 ∨ yᵢ = -1
; |{i | yᵢ = -1}| ≤ 1
; ∏(yᵢ) = ∏(xᵢ)
```

This notation captures:

1. Input/output tensor shapes and dimensions
2. Mathematical constraints on dimensions
3. Runtime error conditions

### Example 1: torch.tensor

Takes a Python array as input and returns a tensor with the dimensions and shape of that array. The dimensions represent the number of axes (rank) while the shape indicates the size along each dimension.

**Parameters:**

- `data` (`array_like`): Initial data for the tensor. Can be a list, tuple, NumPy ndarray, scalar, and other types.

**Type Signature:**

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

**Type Signature:**

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

**Type Signature:**

```python
;; reshape :: Tensor ([x_1, x_2, ... , x_n], dim=n) Shape(y_1, y_2, ..., y_m) -> Tensor ([y_1, y_2, ..., y_m], dim=m)
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

**Type Signature:**

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

**Type Signature:**

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

### Example 6: torch.squeeze

Removes dimensions of size 1 from the tensor.

When dim is given, a squeeze operation is done only in the given dimension(s).
Example: If input is of shape (A×1×B), then:

- squeeze(input, 0) leaves the tensor unchanged
- squeeze(input, 1) will squeeze the tensor to shape (A×B)

**Parameters:**

- `input` (`Tensor`): The input tensor.
- `dim` (`int`, optional): If given, the input will be squeezed only in this dimension.

**Constraints:**

- If dim is specified, it must be in range [-n, n - 1] where n is the number of dimensions

```python
# IndexError - Dimension out of range
x = torch.zeros(2, 1, 2)  # Shape (2,1,2)
torch.squeeze(x, dim=3)  # dim 3 is out of range [-3, 2]
>>> IndexError: Dimension out of range (expected to be in range of [-3, 2], but got 3)
```

**Type Signature:**

```python
;; squeeze :: Tensor([x_1, x_2, ..., x_n], dim=n) -> Tensor([y_1, y_2, ..., y_m], dim=m)
; where m ≤ n and y_i = x_j where x_j ≠ 1
; constraints:
; ∀i ∈ [1..n]: x_i = 1 ∨ x_i = y_j for some j

# Dimension and Shape - before squeeze
x = torch.zeros(2, 1, 2, 1, 2)  # Shape (2,1,2,1,2)
x.dim() # 5
x.size() # torch.Size([2, 1, 2, 1, 2])

# Dimension and Shape - after squeeze
x = torch.squeeze(x)
x.dim() # 3
x.size() # torch.Size([2, 2, 2])  # Removed all dimensions of size 1

# With dim parameter
y = torch.zeros(2, 1, 2, 1, 2)
y = torch.squeeze(y, dim=1)
y.dim() # 4
y.size() # torch.Size([2, 2, 1, 2])  # Only removed dimension at index 1
```

### Example 7: torch.unsqueeze

Adds a dimension of size 1 at the specified position.

**Parameters:**

- `input` (`Tensor`): The input tensor.
- `dim` (`int`): The index at which to insert the singleton dimension.

**Constraints:**

- Dimension must be in range [-n - 1, n] where n is the number of dimensions

```python
# IndexError - Dimension out of range
x = torch.tensor([1, 2, 3, 4])  # Shape (4)
torch.unsqueeze(x, 2)  # dim 2 is out of range [-2, 1]
>>> IndexError: Dimension out of range (expected to be in range of [-2, 1], but got 2)
```

**Type Signature:**

```python
;; unsqueeze :: Tensor([x_1, x_2, ..., x_n], dim=n) Int=m -> Tensor([y_1, y_2, ..., y_{n+1}], dim=n+1)
; where y_i = 1 if i = dim, otherwise y_i = x_j where j = i if i < dim else i-1
; constraints:
; -n-1 ≤ m ≤ n

# Dimension and Shape - before unsqueeze
x = torch.tensor([1, 2, 3, 4])  # Shape (4)
x.dim() # 1
x.size() # torch.Size([4])

# Dimension and Shape - after unsqueeze at position 0
x = torch.unsqueeze(x, 0)
x.dim() # 2
x.size() # torch.Size([1, 4])  # Makes it a row vector

# Dimension and Shape - after unsqueeze at position 1
x = torch.tensor([1, 2, 3, 4])
x = torch.unsqueeze(x, 1)
x.dim() # 2
x.size() # torch.Size([4, 1])  # Makes it a column vector
```

### Example 8: torch.expand

Returns a new view of the self tensor with singleton dimensions expanded to a larger size. Tensor can be also expanded to a larger number of dimensions, and the new ones will be appended at the front. For the new dimensions, the size cannot be set to -1.

Some considerations:

- Passing -1 as the size for a dimension means not changing the size of that dimension.
- This function is a method of the tensor, not a torch function

**Parameters:**

- `\*sizes` (torch.Size or int...) – the desired expanded size

**Constraints:**

- The number of dimensions in the expanded size must be greater than or equal to the number of dimensions in the input tensor

```python
# RuntimeError - Invalid expansion size
x = torch.tensor([[1], [2], [3]])  # Shape (3,1)
x.expand(2, 4)  # Error: dimension 0 must match input size
RuntimeError: The expanded size of the tensor (2) must match the existing size (3) at non-singleton dimension 0.  Target sizes: [2, 4].  Tensor sizes: [3, 1]

```

**Type Signature:**

```python
;; Tensor t =  Tensor([x_1, ..., x_n], dim=n)
;; t.expand ::(y_1, ..., y_m) -> Tensor([y_1, ..., y_m], dim=m)
; constraints:
; ∀i ∈ [1..n]: (x_i = 1 ∧ y_i ≥ 1)
; ∀i ∈ [1..n]: (x_i > 1 ∧ y_i = x_i)
; ∀i ∈ [n+1..m]: y_i ≥ 1 # This one does not produce an error but returns this tensor([])

# Dimension and Shape - before expand
x = torch.tensor([[1], [2], [3]])  # Shape (3,1)
x.dim() # 2
x.size() # torch.Size([3, 1])

# Dimension and Shape - after expand
x = x.expand(3, 4)  # Expands singleton dimension
x.dim() # 2
x.size() # torch.Size([3, 4])

print(x)
>>> tensor([[1, 1, 1, 1],
            [2, 2, 2, 2],
            [3, 3, 3, 3]])
```

### Example 9: torch.stack

Concatenates a sequence of tensors along a new dimension. All tensors must have the same shape.

**Parameters:**

- `tensors` (`sequence of Tensors`): Sequence of tensors to concatenate.
- `dim` (`int`, optional): The dimension to insert. Has to be between 0 and the number of dimensions of input tensors (inclusive).

**Constraints:**

- All input tensors must have the same shape

```python
# RuntimeError - Shapes don't match
x = torch.tensor([1, 2, 3])  # Shape (3)
y = torch.tensor([4, 5])     # Shape (2)
torch.stack((x, y))
>>> RuntimeError: stack expects each tensor to be equal size, but got [3] at entry 0 and [2] at entry 1
```

- Dimension to insert has to be between -n and n -1

```python
# IndexError - Dimension out of range

x = torch.tensor([1, 2, 3]) # Shape (3)
y = torch.tensor([4, 5, 6]) # Shape (3)
torch.stack((x, y), dim=2) # dim 2 is out of range [-2, 1]

>>> IndexError: Dimension out of range (expected to be in range of [-2, 1], but got 2)
```

**Type Signature:**

```python
;; stack :: (Tensor([x_1, ..., x_n], dim=n), ... , Tensor([x_1, ..., x_n], dim=n)) Int=m -> Tensor([x_1, ..., x_{m-1}, k, x_m, ..., x_n], dim=n+1)
; constraints:
; ∀t1,t2 ∈ tensors: t1.shape = t2.shape
; -n <= m <= n -1

# Dimension and Shape - before stack
x = torch.tensor([1, 2, 3])  # Shape (3)
y = torch.tensor([4, 5, 6])  # Shape (3)
z = torch.tensor([7, 8, 9])  # Shape (3)

# Dimension and Shape - after stack
t = torch.stack((x, y, z), dim=0)
t.dim() # 2
t.size() # torch.Size([3, 3])

print(t)
>>> tensor([[1, 4, 7],
            [2, 5, 8],
            [3, 6, 9]])
```

### Example 10: torch.transpose

Returns a tensor that is a transposed version of input. The given dimensions dim0 and dim1 are swapped.

**Parameters:**

- `input` (`Tensor`): The input tensor.
- `dim0` (`int`): The first dimension to be transposed
- `dim1` (`int`): The second dimension to be transposed

**Constraints:**

- Dimensions must be in range [-n, n - 1] where n is the number of dimensions

```python
# IndexError - Dimension out of range
x = torch.randn(2, 3)  # Shape (2,3)
torch.transpose(x, 0, 2)  # dim 2 is out of range [-2, 1]
>>> IndexError: Dimension out of range (expected to be in range of [-2, 1], but got 2)
```

**Type Signature:**

```python
;; transpose :: Tensor([x_1, ... dim0, ..., dim1, ..., x_n], dim=n) Int=dim0 Int=dim1 -> Tensor([y_1, ...,dim1, ..., dim0, ... y_n], dim=n)
; constraints:
; -n ≤ dim0, dim1 < n
; dim0 ≠ dim1 # This is allowed but nonsense

# Dimension and Shape - before transpose
x = torch.randn(2, 3, 4)  # Shape (2,3,4)
x.dim() # 3
x.size() # torch.Size([2, 3, 4])

# Dimension and Shape - after transpose
t = torch.transpose(x, 0, 1)
t.dim() # 3
t.size() # torch.Size([3, 2, 4])

print(t)
>>> tensor([[[0.1, 0.2, 0.3, 0.4],
             [0.5, 0.6, 0.7, 0.8]],
            [[0.9, 1.0, 1.1, 1.2],
             [1.3, 1.4, 1.5, 1.6]],
            [[1.7, 1.8, 1.9, 2.0],
             [2.1, 2.2, 2.3, 2.4]]])
```
