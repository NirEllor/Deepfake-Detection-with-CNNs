import torch.nn as nn
from torch.utils.data import Dataset
from helpers import *
import pandas as pd
import numpy as np

LEARNING_RATES = [1, 0.01, 0.001, 0.00001]
SPECIAL_EPOCH = 100
BATCH_SIZE = [1, 16, 128, 1024]
EPOCHS = [1, 10, 50, 50]


class EuropeDataset(Dataset):
    def __init__(self, csv_file):
        """
        Args:
            csv_file (string): Path to the CSV file with annotations.
        """
        # Load the data into tensors
        file_pre_data, _ = read_data_demo(csv_file)

        # Convert NumPy arrays to PyTorch tensors with the correct data types
        self.features = torch.from_numpy(file_pre_data[:, 1:3]).float()  # Features as float
        self.labels = torch.from_numpy(file_pre_data[:, -1]).long()      # Labels as long (int64)

        # Dataset properties
        self.data_set_size = len(self.features)
        self.data_dimension = self.features.shape[1]
        self.num_classes = len(torch.unique(self.labels))


    def __len__(self):
        """Re☺turns the total number of samples in the dataset."""
        #### YOUR CODE HERE ####
        return self.data_set_size

    def __getitem__(self, idx):
        """☺
        Args:
            idx (int): Index of the data row
        
        Returns:
            dictionary or list corresponding to a feature tensor, and it's corresponding label tensor
        """
        #### YOUR CODE HERE ####
        return self.features[idx], self.labels[idx]
    

class MLP(nn.Module):
    def __init__(self, num_hidden_layers, hidden_dim, output_dim, extra_batch_norm=False):
        super(MLP, self).__init__()
        #### YOUR CODE HERE ####
        """
        Args:
            num_hidden_layers (int): The number of hidden layers, in total you'll have an extra layer at the end,
            from hidden_dim to output_dim
            hidden_dim (int): The hidden layer dimension
            output_dim (int): The output dimension, should match the number of classes in the dataset
        """
        # Create a list to hold the layers
        if extra_batch_norm:
            layers = [nn.Linear(2, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU()]
        else:
            layers = [nn.Linear(2, hidden_dim), nn.ReLU()]
        # First layer: from input_dim to hidden_dim

        # Add num_hidden_layers hidden layers
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if extra_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())  # Activation function

        # Final layer: from hidden_dim to output_dim
        layers.append(nn.Linear(hidden_dim, output_dim))

        # Use nn.Sequential to combine all layers
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        #### YOUR CODE HERE ####
       return self.network(x)


def train(train_dataset, val_dataset, test_dataset, model, lr=0.001, epochs=50, batch_size=256):

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    validation_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1024, shuffle=False, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1024, shuffle=False, num_workers=0)

    len_train_loader = len(train_loader)
    len_validation_loader = len(validation_loader)
    len_test_loader = len(test_loader)

    # Initialize criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_accs = []
    val_accs = []
    test_accs = []
    train_losses = []
    val_losses = []
    test_losses = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    for ep in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        batch_accs = []  # To store accuracy for each batch

        for inputs, labels in train_loader:
            (inputs,
             labels) = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Track training loss
            running_loss += loss.item()

            # Compute batch accuracy
            _, predicted = torch.max(outputs, dim=1)
            batch_acc = (predicted == labels).float().mean().item()
            batch_accs.append(batch_acc)

        avg_train_loss = running_loss / len_train_loader
        avg_train_acc = np.mean(batch_accs)  # Average of batch accuracies
        train_losses.append(avg_train_loss)
        train_accs.append(avg_train_acc)

        # Validation phase
        model.eval()
        running_val_loss = 0.0
        batch_val_accs = []  # To store accuracy for each batch

        with torch.no_grad():
            for inputs, labels in validation_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                val_loss = criterion(outputs, labels)
                running_val_loss += val_loss.item()

                # Compute batch accuracy
                _, predicted = torch.max(outputs, dim=1)
                batch_val_acc = (predicted == labels).float().mean().item()
                batch_val_accs.append(batch_val_acc)

        avg_val_loss = running_val_loss / len_validation_loader
        avg_val_acc = np.mean(batch_val_accs)
        val_losses.append(avg_val_loss)
        val_accs.append(avg_val_acc)

        # Test phase (optional)
        running_test_loss = 0.0
        batch_test_accs = []  # To store accuracy for each batch

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                test_loss = criterion(outputs, labels)
                running_test_loss += test_loss.item()

                # Compute batch accuracy
                _, predicted = torch.max(outputs, dim=1)
                batch_test_acc = (predicted == labels).float().mean().item()
                batch_test_accs.append(batch_test_acc)

        avg_test_loss = running_test_loss / len_test_loader
        avg_test_acc = np.mean(batch_test_accs)
        test_losses.append(avg_test_loss)
        test_accs.append(avg_test_acc)

        # Print epoch statistics
        print('Epoch {:}, Train Loss: {:.3f}, Val Loss: {:.3f}, Test Loss: {:.3f}, '
              'Train Acc: {:.3f}, Val Acc: {:.3f}, Test Acc: {:.3f}'.format(
            ep + 1, avg_train_loss, avg_val_loss, avg_test_loss, avg_train_acc, avg_val_acc, avg_test_acc
        ))
    return model, train_accs, val_accs, test_accs, train_losses, val_losses, test_losses

def test_model_arbitrary_lr(output_dimension, train_dataset, val_dataset, test_dataset, epochs=50, batch_size=256, extra_batch_norm=False, losses_title="Losses", accuracies_title="Accuracies" ,lr=0.001):
    model = MLP(6, 16, output_dimension, extra_batch_norm=extra_batch_norm)
    model, train_accs, val_accs, test_accs, train_losses, val_losses, test_losses = train(train_dataset, val_dataset,
    test_dataset, model,lr=lr,epochs=epochs,batch_size=batch_size)
    plot_model_loss_acc(train_losses, val_losses, test_losses, train_accs, val_accs, test_accs, model, losses_title=losses_title, accuracies_title=accuracies_title)


def plot_model_loss_acc(train_losses, val_losses, test_losses, train_accs, val_accs, test_accs, model, losses_title="Losses", accuracies_title="Accuracies",):
    plt.figure()
    plt.plot(train_losses, label='Train', color='red')
    plt.plot(val_losses, label='Val', color='blue')
    plt.plot(test_losses, label='Test', color='green')
    plt.title(losses_title)
    plt.legend()
    plt.show()
    plt.figure()
    plt.plot(train_accs, label='Train', color='red')
    plt.plot(val_accs, label='Val', color='blue')
    plt.plot(test_accs, label='Test', color='green')
    plt.title(accuracies_title)
    plt.legend()
    plt.show()
    # train_data = pd.read_csv('train.csv')
    # val_data = pd.read_csv('validation.csv')
    test_data = pd.read_csv('test.csv')
    # plot_decision_boundaries(model, test_data[['long', 'lat']].values, test_data['country'].values,
    #                          'Decision Boundaries', implicit_repr=False)


def test_learning_rates_losses(output_dimension):
    validation_accuracies = []

    for lr in LEARNING_RATES:
        print(f"------------------------- learning rate is {lr:.5f}-------------------------------")
        print()
        model_lr = MLP(6, 16, output_dimension)
        val_accuracy = train(train_dataset,val_dataset,test_dataset, model_lr,lr=lr,epochs=50,batch_size=256)[5]
        validation_accuracies.append(val_accuracy)
        print()
    plt.figure()
    plt.plot(validation_accuracies[0], label=LEARNING_RATES[0], color='red')
    plt.plot(validation_accuracies[1], label=LEARNING_RATES[1], color='blue')
    plt.plot(validation_accuracies[2], label=LEARNING_RATES[2], color='green')
    plt.plot(validation_accuracies[3], label=f"{LEARNING_RATES[3]:.5f}", color='yellow')
    plt.title('Losses per learning rate in the validation set')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def test_learning_rates_accuracies(output_dimension):
    validation_losses = []
    for index, batch in enumerate(BATCH_SIZE):
        print(f"------------------------- ({batch},{EPOCHS[index]})-------------------------------")
        print()
        model_lr = MLP(6, 16, output_dimension)
        val_losses = train(train_dataset,val_dataset,test_dataset, model_lr,lr=0.001,epochs=EPOCHS[index],batch_size=batch)[2]
        validation_losses.append(val_losses)
        print()
    plt.figure()
    plt.plot(validation_losses[0], label=f"({BATCH_SIZE[0]},1)", color='red')
    plt.plot(validation_losses[1], label=f"({BATCH_SIZE[1]},1)", color='blue')
    plt.plot(validation_losses[2], label=f"({BATCH_SIZE[2]},50)", color='green')
    plt.plot(validation_losses[3], label=f"({BATCH_SIZE[3]},50)", color='yellow')
    plt.title('Accuracies per (Batch, epochs) in the validation set')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    # seed for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)
    train_dataset = EuropeDataset('train.csv')
    val_dataset = EuropeDataset('validation.csv')
    test_dataset = EuropeDataset('test.csv')

    #### YOUR CODE HERE #####
    output_dim = len(train_dataset.labels.unique())

    # test_model_arbitrary_lr(output_dim, train_dataset, val_dataset, test_dataset, losses_title="Losses regular model")
    # test_learning_rates_losses(output_dim)
    # test_model_arbitrary_lr(output_dim, train_dataset, val_dataset, test_dataset, epochs=100)
    # test_model_arbitrary_lr(output_dim, train_dataset, val_dataset, test_dataset, extra_batch_norm=True, losses_title="Losses modified model")
    test_learning_rates_accuracies(output_dim)