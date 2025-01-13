import torch.nn as nn
import os
import torch
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms
import torchvision
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


class ResNet18(nn.Module):
    def __init__(self, pretrained=False, probing=False):
        super(ResNet18, self).__init__()
        if pretrained:
            weights = ResNet18_Weights.IMAGENET1K_V1
            self.transform = ResNet18_Weights.IMAGENET1K_V1.transforms()
            self.resnet18 = resnet18(weights=weights)
        else:
            self.transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
            self.resnet18 = resnet18()

        in_features_dim = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Identity()
        # Freeze parameters if probing
        if probing:
            for name, param in self.resnet18.named_parameters():
                param.requires_grad = False
        self.logistic_regression = nn.Linear(in_features_dim, 1)

    def forward(self, x):
        features = self.resnet18(x)
        ### YOUR CODE HERE ###
        logits = self.logistic_regression(features)
        return logits


def get_loaders(path, transform, batch_size):
    """
    Get the data loaders for the train, validation and test sets.
    :param path: The path to the 'whichfaceisreal' directory.
    :param transform: The transform to apply to the images.
    :param batch_size: The batch size.
    :return: The train, validation and test data loaders.
    """
    train_set = torchvision.datasets.ImageFolder(root=os.path.join(path, 'train'), transform=transform)
    val_set = torchvision.datasets.ImageFolder(root=os.path.join(path, 'val'), transform=transform)
    test_set = torchvision.datasets.ImageFolder(root=os.path.join(path, 'test'), transform=transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def compute_accuracy(model, data_loader, device):
    """
    Compute the accuracy of the model on the data in data_loader
    :param model: The model to evaluate.
    :param data_loader: The data loader.
    :param device: The device to run the evaluation on.
    :return: The accuracy of the model on the data in data_loader
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in data_loader:
            imgs, labels = imgs.to(device), labels.to(device).float()
            outputs = model(imgs).squeeze()
            predictions = (torch.sigmoid(outputs) > 0.5).int()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total
    return accuracy


def run_training_epoch(model, criterion, optimizer, train_loader, device):
    """
    Run a single training epoch
    :param model: The model to train
    :param criterion: The loss function
    :param optimizer: The optimizer
    :param train_loader: The data loader
    :param device: The device to run the training on
    :return: The average loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    total_batches = len(train_loader)
    for (imgs, labels) in tqdm(train_loader, total=len(train_loader)):
        ### YOUR CODE HERE ###
        imgs, labels = imgs.to(device), labels.to(device).float()

        # Forward pass
        outputs = model(imgs).squeeze()
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / total_batches
    return avg_loss

OPTIONS = [[False, False], [True, True], [True, False]]
# Set the random seed for reproducibility
torch.manual_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for (pretrained, probing) in OPTIONS:
    print(f"--------------pretrained = {pretrained}, probing = {probing}---------------")

    batch_size = 32
    num_of_epochs = 1
    learning_rates = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    path = 'whichfaceisreal'  # For example '/cs/usr/username/whichfaceisreal/'

    for lr in learning_rates:
        model = ResNet18(pretrained=pretrained, probing=probing)
        transform = model.transform
        train_loader, val_loader, test_loader = get_loaders(path, transform, batch_size)
        model = model.to(device)

        ### Define the loss function and the optimizer
        criterion = torch.nn.BCEWithLogitsLoss()
        ### Train the model
        if probing:
            # Only optimize the final linear layer
            optimizer = torch.optim.Adam(model.logistic_regression.parameters(), lr=lr)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        for epoch in range(num_of_epochs):
            # Run a training epoch
            loss = run_training_epoch(model, criterion, optimizer, train_loader, device)
            # Compute the accuracy
            train_acc = compute_accuracy(model, train_loader, device)
            # Compute the validation accuracy
            val_acc = compute_accuracy(model, val_loader, device)
            print(f' Learning Rate: {lr}: Epoch {epoch + 1}/{num_of_epochs}, Loss: {loss:.4f}, Val accuracy: {val_acc:.4f}')
            # Stopping condition
            ### YOUR CODE HERE ###
            if val_acc > 0.95:
                print("Stopping early as validation accuracy exceeded 95%")
                break
        # Compute the test accuracy
        test_acc = compute_accuracy(model, test_loader, device)
        print(f'Learning Rate of {lr} Test accuracy: {test_acc:.4f}')

# Load pretrained ResNet18 and remove the final layer (fc)
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1
        self.resnet18 = resnet18(weights=weights)
        self.resnet18.fc = nn.Identity()  # Replace fc layer with Identity to extract features

    def forward(self, x):
        return self.resnet18(x)


def extract_features(loader, model, device):
    """
    Extract features from the dataset using the pretrained ResNet18 model.
    :param loader: DataLoader for the dataset (train/val/test)
    :param model: Pretrained feature extractor model
    :param device: Device to run the model on (CPU/GPU)
    :return: Tuple of (features, labels) as NumPy arrays
    """
    features = []
    labels = []
    with torch.no_grad():
        for imgs, lbls in tqdm(loader, total=len(loader), desc='Extracting features'):
            imgs = imgs.to(device)
            lbls = lbls.cpu().numpy()
            feats = model(imgs).cpu().numpy()  # Extract features and convert to NumPy
            features.append(feats)
            labels.append(lbls)

    # Concatenate all batches into a single array
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    return features, labels


# Initialize the feature extractor
model = FeatureExtractor().to(device)
model.eval()  # Set the model to evaluation mode

# Define transformations and data loaders
transform = ResNet18_Weights.IMAGENET1K_V1.transforms()
batch_size = 32
path = 'whichfaceisreal'  # Example path to your dataset
train_loader, val_loader, test_loader = get_loaders(path, transform, batch_size)

# Extract features and labels for train, validation, and test sets
train_features, train_labels = extract_features(train_loader, model, device)
val_features, val_labels = extract_features(val_loader, model, device)
test_features, test_labels = extract_features(test_loader, model, device)

# Train a logistic regression model using sklearn
clf = LogisticRegression(max_iter=1000)
clf.fit(train_features, train_labels)

# Evaluate on the validation set
val_predictions = clf.predict(val_features)
val_accuracy = accuracy_score(val_labels, val_predictions)
print(f'Validation Accuracy: {val_accuracy:.4f}')

# Evaluate on the test set
test_predictions = clf.predict(test_features)
test_accuracy = accuracy_score(test_labels, test_predictions)
print(f'Test Accuracy: {test_accuracy:.4f}')

# Helper function to get predictions
def get_predictions(model, data_loader, device):
    """
    Get predictions from a model on the dataset provided by data_loader.
    :param model: The model to use for prediction.
    :param data_loader: DataLoader for the dataset.
    :param device: Device to run the model on (CPU/GPU).
    :return: Predictions and true labels as NumPy arrays.
    """
    model.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        for imgs, labels in data_loader:
            imgs = imgs.to(device)
            outputs = model(imgs).squeeze()
            preds = (torch.sigmoid(outputs) > 0.5).int().cpu().numpy()
            predictions.extend(preds)
            true_labels.extend(labels.numpy())

    return np.array(predictions), np.array(true_labels)


# Helper function to visualize samples
def visualize_samples(indices, data_loader, title):
    """
    Visualize samples given their indices.
    :param indices: List of indices to visualize.
    :param data_loader: DataLoader for the dataset.
    :param title: Title for the plot.
    """
    plt.figure(figsize=(15, 5))

    for i, idx in enumerate(indices[:5]):  # Visualize only the first 5 samples
        img, label = data_loader.dataset[idx]
        plt.subplot(1, 5, i + 1)
        plt.imshow(img.permute(1, 2, 0))  # Convert from (C, H, W) to (H, W, C)
        plt.title(f"Label: {label}")
        plt.axis('off')

    plt.suptitle(title)
    plt.show()


# Best model: Linear probing with lr=0.0001
best_model = ResNet18(pretrained=True, probing=False).to(device)
best_optimizer = torch.optim.Adam(best_model.logistic_regression.parameters(), lr=0.0001)
best_model.eval()

# Worst model: ResNet18 trained from scratch with lr=0.1
worst_model = ResNet18(pretrained=True, probing=False).to(device)
worst_optimizer = torch.optim.Adam(worst_model.parameters(), lr=0.1)
worst_model.eval()

# Define transformations and data loaders
transform = ResNet18_Weights.IMAGENET1K_V1.transforms()
batch_size = 32
path = 'whichfaceisreal'  # Example path to your dataset
train_loader, val_loader, test_loader = get_loaders(path, transform, batch_size)

# Get predictions from both models
best_predictions, true_labels = get_predictions(best_model, test_loader, device)
worst_predictions, _ = get_predictions(worst_model, test_loader, device)

# Find indices where the best model was correct and the worst model was incorrect
correct_indices = np.where(best_predictions == true_labels)[0]  # Correctly classified by best model
misclassified_indices = np.where(worst_predictions != true_labels)[0]  # Misclassified by worst model

# Find common indices
common_indices = np.intersect1d(correct_indices, misclassified_indices)

# Visualize 5 of these samples
visualize_samples(common_indices, test_loader,
                  title="Samples correctly classified by best model but misclassified by worst model")
