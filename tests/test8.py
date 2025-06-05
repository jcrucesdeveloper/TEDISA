import torch
import torch.nn as nn

# Create a network with advanced layers
class AdvancedNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Convolutional layers with different configurations
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        
        # Advanced normalization layers
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Advanced activation functions
        self.activation = nn.Mish()
        
        # Advanced pooling
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layers with different configurations
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 10)
        
        # Advanced regularization
        self.dropout = nn.Dropout2d(0.3)
        
    def forward(self, x):
        # Convolutional block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        # Convolutional block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        # Convolutional block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        # Global pooling
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        
        return x

# Create input data
batch_size = 2
input_data = torch.randn(batch_size, 3, 32, 32)
print("Input shape:", input_data.shape)

# Initialize model
model = AdvancedNN()
print("\nModel architecture:")
print(model)

# Forward pass
output = model(input_data)
print("\nOutput shape:", output.shape)

# Different loss functions
print("\nDifferent loss functions:")
target = torch.tensor([0, 1])
print("Cross Entropy Loss:", nn.CrossEntropyLoss()(output, target))
print("MSE Loss:", nn.MSELoss()(output, torch.randn_like(output)))
print("L1 Loss:", nn.L1Loss()(output, torch.randn_like(output)))
print("Smooth L1 Loss:", nn.SmoothL1Loss()(output, torch.randn_like(output)))

# Different optimizers
print("\nDifferent optimizers:")
optimizer1 = torch.optim.SGD(model.parameters(), lr=0.01)
optimizer2 = torch.optim.Adam(model.parameters(), lr=0.001)
optimizer3 = torch.optim.RMSprop(model.parameters(), lr=0.001)
optimizer4 = torch.optim.AdamW(model.parameters(), lr=0.001)

# Learning rate schedulers
print("\nLearning rate schedulers:")
scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=30, gamma=0.1)
scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=10)
scheduler3 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer3, mode='min', factor=0.1)

# Weight initialization
print("\nWeight initialization:")
def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)

model.apply(init_weights)
print("Weights initialized with Kaiming and Xavier methods") 