
# Import PyTorch, dataset tools, and our CNN architecture
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import SimpleCNN
from sklearn.metrics import confusion_matrix, classification_report  # Evaluation tools
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

# Load the Fashion-MNIST test dataset
test_dataset = datasets.FashionMNIST(
    root="./data",
    train=False,  # Use images the model didn't train on
    download=True,
    transform=transforms.ToTensor()  # Convert images to tensors
)

# Organize test images into batches
test_loader = DataLoader(
    test_dataset,
    batch_size=32,  # Evaluate 32 images at a time
    shuffle=False  # Keep a consistent evaluation order
)

# Recreate our CNN and restore its trained weights
model = SimpleCNN()
model.load_state_dict(
    torch.load("fashion_cnn.pth", map_location="cpu", weights_only=True)
)

model.eval()  # Switch to evaluation mode

# Create lists to store correct labels and model predictions
all_true = []  # Actual clothing categories
all_pred = []  # Categories predicted by our CNN
all_probs = []  # Store class probability scores for ROC-AUC

# Evaluate every test image
with torch.no_grad():  # No gradient calculations needed
    for images, labels in test_loader:
        outputs = model(images)  # Generate class scores
        predictions = outputs.argmax(dim=1)  # Find highest-scoring class
        probabilities = torch.softmax(outputs, dim=1)  # Convert logits to class probabilities
        all_probs.extend(probabilities.tolist())  # Store scores for all 10 classes

        all_true.extend(labels.tolist())  # Store actual labels
        all_pred.extend(predictions.tolist())  # Store predictions

# Verify that we evaluated the full dataset
print("Actual labels collected:", len(all_true))
print("Predictions collected:", len(all_pred))

print("First 5 actual labels:", all_true[:5])
print("First 5 predictions:", all_pred[:5])

# Build our confusion matrix
cm = confusion_matrix(all_true, all_pred)  # Compare actual labels against predictions

print("\nConfusion Matrix:")
print(cm)  # Rows = actual categories, columns = predicted categories

# Calculate precision, recall, and F1 for every clothing category
report = classification_report(
    all_true,
    all_pred,
    target_names=test_dataset.classes  # Show clothing names instead of numbers
)

print("\nClassification Report:")
print(report)

# Calculate multiclass ROC-AUC
auc = roc_auc_score(
    all_true,
    all_probs,
    multi_class="ovr",  # Compare each clothing category against all others
    average="macro"  # Give each clothing category equal importance
)

print(f"\nROC-AUC: {auc:.4f}")  # Display the overall score
