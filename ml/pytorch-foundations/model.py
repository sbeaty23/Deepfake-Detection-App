# Import PyTorch neural-network tools
import torch
from torch import nn
import torch.nn.functional as F

# Define the CNN architecture
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)  # Grayscale image → 16 feature maps
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # 16 → 32 feature maps

        self.pool = nn.MaxPool2d(2, 2)  # Halve spatial dimensions
        self.fc1 = nn.Linear(32 * 7 * 7, 128)  # Convert flattened features to 128 values
        self.fc2 = nn.Linear(128, 10)  # Produce scores for 10 clothing categories

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # First convolution block
        x = self.pool(F.relu(self.conv2(x)))  # Second convolution block

        x = torch.flatten(x, 1)  # Flatten each image's feature maps
        x = F.relu(self.fc1(x))  # Apply first fully connected layer
        x = self.fc2(x)  # Produce final logits

        return x  # Return category scores
