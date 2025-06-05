import torch

# Create a simple model
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    
    def forward(self, x):
        return self.linear(x)

# Create model and input
model = SimpleModel()
x = torch.randn(3, 2)
print("Input shape:", x.shape)

# Basic forward pass
print("\nBasic forward pass:")
output = model(x)
print("Output shape:", output.shape)

# Compile the model
print("\nCompiling model:")
compiled_model = torch.compile(model)
print("Compiled model type:", type(compiled_model))

# Forward pass with compiled model
print("\nForward pass with compiled model:")
compiled_output = compiled_model(x)
print("Compiled output shape:", compiled_output.shape)

# Compare outputs
print("\nOutput comparison:")
print("Original output:", output)
print("Compiled output:", compiled_output)
print("Are outputs equal?", torch.allclose(output, compiled_output))

# Performance comparison
import time

def time_execution(func, *args, **kwargs):
    start = time.time()
    result = func(*args, **kwargs)
    end = time.time()
    return result, end - start

print("\nPerformance comparison:")
_, original_time = time_execution(model, x)
_, compiled_time = time_execution(compiled_model, x)
print(f"Original model time: {original_time:.6f}s")
print(f"Compiled model time: {compiled_time:.6f}s")
print(f"Speedup: {original_time/compiled_time:.2f}x") 