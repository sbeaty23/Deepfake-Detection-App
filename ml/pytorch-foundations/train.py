import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch import nn #nn contains the neural-network building blocks.
from model import SimpleCNN  # Import our CNN architecture

transform = transforms.ToTensor()

train_dataset = datasets.FashionMNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.FashionMNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

print("Training images:", len(train_dataset))
print("Testing images:", len(test_dataset))

image, label = train_dataset[0]

print("Image shape:", image.shape)
print("Label:", label)
print("Class name:", train_dataset.classes[label])

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

images, labels = next(iter(train_loader))

print("Batch image shape:", images.shape)
print("Batch label shape:", labels.shape)

model = SimpleCNN() #Creates an instance of the CNN architecture you defined.

outputs = model(images) #Passes your batch of 32 images through the network. This is called a forward pass.

print("Model output shape:", outputs.shape) #Tells us the dimensions of the predictions.
print("First image scores:", outputs[0]) #Shows the 10 raw class scores (called logits) for the first image in the batch.

# Define our loss function
criterion = nn.CrossEntropyLoss() #common loss function for classification problems involving multiple categories.

# Define our optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) #model.parameters() tells Adam which model weights and biases it is allowed to update. lr is learning rate.

print("Loss function:", criterion)
print("Optimizer:", optimizer.__class__.__name__)

# Train our CNN
epochs = 3  # Number of times we go through the full training dataset

for epoch in range(epochs):  # Repeat training for each epoch
    model.train()  # Put the model in training mode
    running_loss = 0.0  # Track total loss for this epoch

    for images, labels in train_loader:  # Get one batch of images and labels
        optimizer.zero_grad()  # Clear gradients from the previous batch

        outputs = model(images)  # Make predictions for the batch
        loss = criterion(outputs, labels)  # Compare predictions to correct labels

        loss.backward()  # Calculate gradients using backpropagation
        optimizer.step()  # Update model weights using those gradients

        running_loss += loss.item()  # Add batch loss to running total

    average_loss = running_loss / len(train_loader)  # Average loss across all batches

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {average_loss:.4f}")  # Display progress

# Evaluate the trained CNN
model.eval()  # Switch model into evaluation mode

correct = 0  # Track correctly classified images
total = 0  # Track total images evaluated

test_loader = DataLoader(
    test_dataset,
    batch_size=32,  # Evaluate 32 images per batch
    shuffle=False  # No need to shuffle evaluation images
)

with torch.no_grad():  # Disable gradient tracking during evaluation
    for images, labels in test_loader:
        outputs = model(images)  # Predict using trained weights

        _, predicted = torch.max(outputs, 1)  # Select class with highest score

        total += labels.size(0)  # Count images in this batch
        correct += (predicted == labels).sum().item()  # Count correct predictions

accuracy = 100 * correct / total  # Calculate accuracy percentage

print(f"Test Accuracy: {accuracy:.2f}%")  # Display evaluation result

# Save the trained CNN
torch.save(model.state_dict(), "fashion_cnn.pth")  # Save the learned weights and biases

print("Model saved successfully!")  # Confirm checkpoint creation
