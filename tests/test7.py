import torch
import torch.nn as nn

# Create a more complex neural network
class ComplexNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(16 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return self.softmax(x)

# Create input data
batch_size = 2
input_data = torch.randn(batch_size, 1, 28, 28)
print("Input shape:", input_data.shape)

# Initialize model
model = ComplexNN()
print("\nModel architecture:")
print(model)

# Forward pass
output = model(input_data)
print("\nOutput shape:", output.shape)
print("Output probabilities:", output)

# Loss functions
criterion = nn.CrossEntropyLoss()
target = torch.tensor([0, 1])
loss = criterion(output, target)
print("\nCross Entropy Loss:", loss)

# Backward pass
loss.backward()
print("\nGradients after backward pass:")
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name} grad shape:", param.grad.shape)

# Different activation functions
x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
print("\nDifferent activation functions:")
print("Input:", x)
print("ReLU:", nn.ReLU()(x))
print("Leaky ReLU:", nn.LeakyReLU(0.1)(x))
print("ELU:", nn.ELU()(x))
print("Sigmoid:", nn.Sigmoid()(x))
print("Tanh:", nn.Tanh()(x))
print("GELU:", nn.GELU()(x))

# Different pooling operations
pool_input = torch.randn(1, 1, 4, 4)
print("\nPooling operations:")
print("Input shape:", pool_input.shape)
print("Max Pool:", nn.MaxPool2d(2)(pool_input))
print("Avg Pool:", nn.AvgPool2d(2)(pool_input))
print("Adaptive Avg Pool:", nn.AdaptiveAvgPool2d((2, 2))(pool_input)) 