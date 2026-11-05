import torch  # Provides tensors and PyTorch operations

from torch import nn  # Provides neural network layers

# Import ResNet50 and its pretrained weight configurations
from torchvision.models import resnet50, ResNet50_Weights

# Choose the pretrained weights
weights = ResNet50_Weights.DEFAULT  # Use recommended ImageNet pretrained weights

# Create our ResNet50 model
model = resnet50(weights=weights)  # Build the network and load learned weights

# Examine its original output layer
print("Original final layer:", model.fc)  # Display the classification layer

# Freeze the pretrained ResNet50 layers
for parameter in model.parameters():
    parameter.requires_grad = False  # Prevent these weights from changing during training

# Replace the original 1000-class output layer
model.fc = nn.Linear(model.fc.in_features, 2)  # New layer predicts real or fake

# Check the modified network
print("New final layer:", model.fc)

# Show which parameters can still be trained
for name, parameter in model.named_parameters():
    if parameter.requires_grad:
        print("Trainable:", name)

# Test our modified ResNet50 with a sample input
model.eval()  # Switch to evaluation mode

sample_image = torch.randn(1, 3, 224, 224)  # Create a dummy RGB image

with torch.no_grad():  # Disable gradient calculations
    outputs = model(sample_image)  # Pass the image through ResNet50

print("Output shape:", outputs.shape)  # Display prediction dimensions
print("Output scores:", outputs)  # Display the two raw classification scores
