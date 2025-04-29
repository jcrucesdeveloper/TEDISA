# Type System first iteration

We need to capture technical requirements for implementing a type system that first captures a Tensor's dimensions and potentially its shape. This will use simple functions from the [PyTorch documentation](https://pytorch.org/docs/stable/index.html). The goal is capture the tensor's shape and dimensions in a type.

### First Approach to Typing Tensor Dimensions and Shapes

The following examples use PyTorch methods that create or modify tensor dimensions and shapes. This represents an initial approach to developing a type system for tracking these properties.

### Example 1: torch.tensor

This is the constructor of tensors, just of the way to create tensor consult more of this in this documentaiton

```python

Takes a Python array as input and returns a tensor with the dimensions and shape of that array. The dimensions represent the number of axes (rank) while the shape indicates the size along each dimension.

;; tensor :: List[Number] -> Tensor ([x_1, x_2, ..., x_n, dim=n])
x = torch.tensor([1, 2, 3])
x.dim() # Returns 1
x.shape # Returns torch.Size([3])
```

### Example 2: torch.flatten

Flattens input by reshaping it into a one-dimensional tensor. The order of elements in input is unchanged.

```python
;; flatten :: Tensor ([x_1, x_2, ... , x_n], dim=n) -> Tensor ([x_1 * x_2 * ... * x_n, dim=1])
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

```python
;; reshape :: Tensor ([x_1, x_2, ... , x_n], dim=n) shape([x^1, x^2, ..., x^m], dim=m) -> Tensor ([x^1, x^2, ..., x^m], dim=m)
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

