# Import required libraries and our CNN architecture
import torch
from torchvision import datasets, transforms
from model import SimpleCNN

# Load the Fashion-MNIST test dataset
transform = transforms.ToTensor()  # Convert images into tensors

test_dataset = datasets.FashionMNIST(
    root="./data",
    train=False,  # Use test images, not training images
    download=True,
    transform=transform
)

# Recreate the CNN and load its saved weights
model = SimpleCNN()  # Create the same network architecture

model.load_state_dict(
    torch.load("fashion_cnn.pth", map_location="cpu", weights_only=True)
)  # Restore weights learned during training

model.eval()  # Switch to evaluation mode

# Select one image from the dataset
image, label = test_dataset[0]  # Get an image and its correct category

image = image.unsqueeze(0)  # Add batch dimension: [1, 1, 28, 28]

# Make a prediction without training
with torch.no_grad():  # Disable gradient calculations
    outputs = model(image)  # Run image through trained CNN
    predicted = outputs.argmax(dim=1).item()  # Find highest-scoring category

# Compare the prediction with the correct answer
print("Actual:", test_dataset.classes[label])
print("Predicted:", test_dataset.classes[predicted])
print("Correct:", predicted == label)
