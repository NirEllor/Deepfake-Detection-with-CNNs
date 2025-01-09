import time
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from helpers import *
import pandas as pd
import numpy as np

WIDTH = "WIDTH"

DEPTH = 'depth'

LEARNING_RATES = [1, 0.01, 0.001, 0.00001]
SPECIAL_EPOCH = 100
BATCH_SIZE = [1, 16, 128, 1024]
EPOCHS = [1, 10, 50, 50]
DEPTHS_WIDTHS = [[1, 16], [2, 16], [6, 16], [10, 16], [6, 8], [6, 32], [6, 64]]
MODELS_WITH_WIDTH_16 = [pair for pair in DEPTHS_WIDTHS if pair[1] == 16]
HIDDEN_LAYERS_WIDTH_16 = [pair[0] for pair in DEPTHS_WIDTHS if pair[1] == 16]
MODELS_WITH_DEPTH_6 = [pair for pair in DEPTHS_WIDTHS if pair[0] == 6]
NUMER_NEURONS_DEPTH_6 = [pair[1] for pair in DEPTHS_WIDTHS if pair[0] == 6]
NUMER_NEURONS_DEPTH_6 = sorted(NUMER_NEURONS_DEPTH_6)


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
    def __init__(self, num_hidden_layers, hidden_dim, output_dim, extra_batch_norm=False, resnet=False, dropout=False, is_sine=False, input_dim=2):
        super(MLP, self).__init__()
        #### YOUR CODE HERE ####
        """
        Args:
            num_hidden_layers (int): The number of hidden layers, in total you'll have an extra layer at the end,
            from hidden_dim to output_dim
            hidden_dim (int): The hidden layer dimension
            output_dim (int): The output dimension, should match the number of classes in the dataset
        """
        # First layer: input_dim -> hidden_dim
        self.resnet = resnet  # Enable residual connections
        self.dropout = dropout
        self.extra_batch_norm = extra_batch_norm

        layers = [nn.Linear(input_dim, hidden_dim)]

        if extra_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU())
        if dropout:
            layers.append(nn.Dropout(p=0.1))  # Dropout after activation

        # Hidden layers: hidden_dim -> hidden_dim
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if extra_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            if dropout:
                layers.append(nn.Dropout(p=0.1))  # Dropout after activation

        # Final layer: hidden_dim -> output_dim
        layers.append(nn.Linear(hidden_dim, output_dim))

        # Combine all layers into a sequential model
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        if not self.resnet:
            return self.network(x)  # Regular forward pass

        # Forward pass with residual connections
        for layer in self.network:
            if isinstance(layer, nn.Linear) and x.size(-1) == layer.in_features:  # Residual condition
                x = x + layer(x) if layer.in_features == layer.out_features else layer(x)
            elif isinstance(layer, nn.ReLU) or isinstance(layer, nn.BatchNorm1d) or isinstance(layer, nn.Dropout):
                x = layer(x)  # Apply activation or batch normalization
        return x


def train(train_dataset, val_dataset, test_dataset, model, lr=0.001, epochs=50, batch_size=256,
          count_iterations=False, track_batch_loss=False, gradient_magnitude=False, clipping=False, schedule=False, regularization=False, to_sine_data=False):

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    validation_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False, num_workers=0)
    if to_sine_data:
        preprocessor = SinusoidalPreprocessor()
        train_transformed_features = preprocessor.transform(train_dataset.features)
        train_dataset = torch.utils.data.TensorDataset(train_transformed_features, train_dataset.labels)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_transformed_features = preprocessor.transform(val_dataset.features)
        val_dataset = torch.utils.data.TensorDataset(val_transformed_features, val_dataset.labels)
        validation_loader = DataLoader(val_dataset, batch_size=1024, shuffle=True)

        test_transformed_features = preprocessor.transform(test_dataset.features)
        test_dataset = torch.utils.data.TensorDataset(test_transformed_features, test_dataset.labels)
        test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=True)


    linear_pos = 1
    features = [model.extra_batch_norm, model.network, model.dropout]
    for feature in features:
        if feature:
            linear_pos += 1
    # Initialize criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    if regularization:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)


    # Initialize learning rate scheduler
    if schedule:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # Decays LR by 0.1 every 10 epochs

    train_accs, val_accs, test_accs = [], [], []
    train_losses, val_losses, test_losses = [], [], []

    # Store gradient magnitudes
    grad_magnitudes = {layer: [] for layer in [0, 30, 60, 90, 95, 99]}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    train_losses_per_batch = [] if track_batch_loss else None


    for ep in range(epochs):
        start_time = time.time()  # Start timing the epoch

        # ----------------------------
        # Training phase
        # ----------------------------
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        cnt = None
        if count_iterations:
            cnt = 0
        batch_count = 0
        # Sum gradients for each layer
        if gradient_magnitude:
            grad_sum = {layer: 0.0 for layer in grad_magnitudes.keys()}

        for inputs, labels in train_loader:
            if count_iterations:
                cnt += 1
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            if clipping:
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           max_norm=8)  # Clip gradients to avoid vanishing/exploding

            # Track gradient magnitudes for selected layers
            if gradient_magnitude:
                for layer_idx in grad_magnitudes.keys():
                # Compute L2 norm squared of the weight gradient
                    weight = torch.pow(model.network[layer_idx * linear_pos].weight.grad.norm(2), 2).item()

                    # Compute L2 norm squared of the bias gradient
                    bias = torch.pow(model.network[layer_idx * linear_pos].bias.grad.norm(2), 2).item()

                    # Sum the L2 norms squared of weight and bias
                    grad_sum[layer_idx] += (weight + bias)

            optimizer.step()
            optimizer.zero_grad()

            # Track loss and accuracy
            running_loss += loss.item() * inputs.size(0)  # Sum loss over the batch
            _, predicted = torch.max(outputs, dim=1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            batch_count += 1

            if track_batch_loss:
                train_losses_per_batch.append(loss.item())  # Track loss after each batch


        avg_train_loss = running_loss / total_samples
        avg_train_acc = correct_predictions / total_samples
        train_losses.append(avg_train_loss)
        train_accs.append(avg_train_acc)
        if count_iterations:
            print(f"Number of iteration for a batch of {batch_size} samples per epoch: {cnt}")

        # Compute mean gradient magnitude for each layer
        if gradient_magnitude:
            for layer_idx in grad_magnitudes.keys():
                grad_magnitudes[layer_idx].append(grad_sum[layer_idx] / batch_count)

        count_iterations = False
        # Step the learning rate scheduler
        if schedule:
            scheduler.step()


        # ----------------------------
        # Validation phase
        # ----------------------------
        model.eval()
        running_val_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        with torch.no_grad():
            for inputs, labels in validation_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                val_loss = criterion(outputs, labels)
                running_val_loss += val_loss.item() * inputs.size(0)

                _, predicted = torch.max(outputs, dim=1)
                correct_predictions += (predicted == labels).sum().item()
                total_samples += labels.size(0)

            avg_val_loss = running_val_loss / total_samples
            avg_val_acc = correct_predictions / total_samples
            val_losses.append(avg_val_loss)
            val_accs.append(avg_val_acc)

        # ----------------------------
        # Test phase (optional)
        # ----------------------------
        with torch.no_grad():
            running_test_loss = 0.0
            correct_predictions = 0
            total_samples = 0
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                test_loss = criterion(outputs, labels)
                running_test_loss += test_loss.item() * inputs.size(0)

                _, predicted = torch.max(outputs, dim=1)
                correct_predictions += (predicted == labels).sum().item()
                total_samples += labels.size(0)

            avg_test_loss = running_test_loss / total_samples
            avg_test_acc = correct_predictions / total_samples
            test_losses.append(avg_test_loss)
            test_accs.append(avg_test_acc)

        end_time = time.time()  # End timing the epoch

        # Print epoch statistics
        print('Epoch {:}, Time: {:.2f}s, Train Loss: {:.3f}, Val Loss: {:.3f}, Test Loss: {:.3f}, '
              'Train Acc: {:.3f}, Val Acc: {:.3f}, Test Acc: {:.3f}'.format(
            ep + 1, end_time - start_time, avg_train_loss, avg_val_loss, avg_test_loss, avg_train_acc, avg_val_acc, avg_test_acc
        ))

    return model, train_accs, val_accs, test_accs, train_losses, val_losses, test_losses, train_losses_per_batch, grad_magnitudes

def test_model_arbitrary_lr(output_dimension, train_dataset, val_dataset, test_dataset, epochs=50, batch_size=512, extra_batch_norm=False, losses_title="Losses", accuracies_title="Accuracies" ,lr=0.001, num_hidden_layers=6,hidden_dim=16,):
    model = MLP(num_hidden_layers, hidden_dim, output_dimension, extra_batch_norm=extra_batch_norm)
    model, train_accs, val_accs, test_accs, train_losses, val_losses, test_losses, _, _ = train(train_dataset, val_dataset,
    test_dataset, model,lr=lr,epochs=epochs,batch_size=batch_size)
    plot_model_loss_acc(train_losses, val_losses, test_losses, train_accs, val_accs, test_accs, losses_title=losses_title, accuracies_title=accuracies_title)


def plot_model_loss_acc(train_losses, val_losses, test_losses, train_accs, val_accs, test_accs, losses_title="Losses", accuracies_title="Accuracies", is_accuracy=False):
    plt.figure()
    plt.plot(train_losses, label='Train', color='red')
    plt.plot(val_losses, label='Val', color='blue')
    plt.plot(test_losses, label='Test', color='green')
    plt.title(losses_title)
    plt.legend()
    plt.show()
    if is_accuracy:
        plt.figure()
        plt.plot(train_accs, label='Train', color='red')
        plt.plot(val_accs, label='Val', color='blue')
        plt.plot(test_accs, label='Test', color='green')
        plt.title(accuracies_title)
        plt.legend()
        plt.show()
    # train_data = pd.read_csv('train.csv')
    # val_data = pd.read_csv('validation.csv')
    # test_data = pd.read_csv('test.csv')
    # plot_decision_boundaries(model, test_data[['long', 'lat']].values, test_data['country'].values,
    #                          'Decision Boundaries', implicit_repr=False)


def test_learning_rates_losses(output_dimension):
    validation_accuracies = []

    for lr in LEARNING_RATES:
        print(f"------------------------- learning rate is {lr:.5f}-------------------------------\n")
        model_lr = MLP(6, 16, output_dimension)
        _,_,_,_,_,val_accuracy,_,_ , _= train(train_dataset,val_dataset,test_dataset, model_lr,lr=lr,epochs=50,batch_size=256)
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

def test_validation_accuracies_per_batch_epoch(output_dimension):
    validation_accuracies = []
    for index, batch in enumerate(BATCH_SIZE):
        print(f"------------------------- (Batch: {batch}, Epochs: {EPOCHS[index]})-------------------------------\n")
        model = MLP(6, 16, output_dimension)
        _, _, val_accuracy, _, _,_,_,_, _ = train(train_dataset,val_dataset,test_dataset, model,lr=0.001,epochs=EPOCHS[index],batch_size=batch, count_iterations = True)
        validation_accuracies.append(val_accuracy)
        print()
    plt.figure()
    plt.plot(validation_accuracies[0], label=f"({BATCH_SIZE[0]},{EPOCHS[0]})", color='red')
    plt.plot(validation_accuracies[1], label=f"({BATCH_SIZE[1]},{EPOCHS[1]})", color='blue')
    plt.plot(validation_accuracies[2], label=f"({BATCH_SIZE[2]},{EPOCHS[2]})", color='green')
    plt.plot(validation_accuracies[3], label=f"({BATCH_SIZE[3]},{EPOCHS[3]})", color='yellow')
    plt.title('Accuracy vs Epoch for each (Batch, epochs) on the validation set')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


def test_train_losses_per_batch_epoch(output_dimension):
    train_losses_per_batch = []  # To store loss curves for each batch size

    for index, batch in enumerate(BATCH_SIZE):
        print(f"------------------------- (Batch: {batch}, Epochs: {EPOCHS[index]})-------------------------------")

        model = MLP(6, 16, output_dimension)

        # Train and collect loss per batch
        _,_,_,_,_,_,_, batch_losses, _ = train(
            train_dataset, val_dataset, test_dataset,
            model, lr=0.001, epochs=EPOCHS[index], batch_size=batch, track_batch_loss=True
        )

        train_losses_per_batch.append(batch_losses)  # Store batch loss curve

    # Plot all loss curves
    plt.figure()
    for i, (loss_curve, batch, epoch) in enumerate(zip(train_losses_per_batch, BATCH_SIZE, EPOCHS)):
        plt.plot(loss_curve, label=f"Batch: {batch}, Epochs: {epoch}")

    plt.title('Loss vs Batch for each (Batch, Epochs) on the train set')
    plt.xlabel('Batch Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def evaluate_MLP_performance(output_dimension):
    final_validation_accuracies = []
    for depth, width in DEPTHS_WIDTHS:
        print(f"------------------------- (Depth: {depth}, width: {width})-------------------------------")

        model = MLP(depth, width, output_dimension)
        _, train_accs, val_accs, test_accs, train_losses, val_losses, test_losses, _, _ = (
            train(train_dataset, val_dataset,test_dataset, model))
        final_validation_accuracies.append(val_accs[-1])

    best_val_model_parameters = DEPTHS_WIDTHS[np.argmax(final_validation_accuracies)]
    worst_val_model_parameters = DEPTHS_WIDTHS[np.argmin(final_validation_accuracies)]
    best_val_accuracy = np.max(final_validation_accuracies)
    worst_val_accuracy = np.min(final_validation_accuracies)
    for index, acc in enumerate(final_validation_accuracies):
        print(f"accuracy of ({DEPTHS_WIDTHS[index][0]}, {DEPTHS_WIDTHS[index][1]}): {acc}")
    print(f"\nThe best model according to the validation accuracy is {best_val_model_parameters} with accuracy {best_val_accuracy}")
    print(f"The worst model according to the validation accuracy is {worst_val_model_parameters} with accuracy {worst_val_accuracy}\n")
    best_val_model = MLP(best_val_model_parameters[0], best_val_model_parameters[1], output_dimension)
    worst_val_model = MLP(worst_val_model_parameters[0], worst_val_model_parameters[1], output_dimension)

    (best_val_model, train_accs_best, val_accs_best, test_accs_best, train_losses_best, val_losses_best,
     test_losses_best, _, _) =  train(train_dataset, val_dataset, test_dataset, best_val_model)

    (worst_val_model, train_accs_worst, val_accs_worst, test_accs_worst, train_losses_worst, val_losses_worst,
     test_losses_worst, _, _) = train(train_dataset, test_dataset, test_dataset, worst_val_model)


    plot_model_loss_acc(train_losses_best, val_losses_best, test_losses_best,
                        train_accs_best, val_accs_best, test_accs_best,
                        losses_title=f"Losses over Best Model: depth {best_val_model_parameters[0]}, width {best_val_model_parameters[1]}",
                        accuracies_title=f"Accuracies over Best Model: depth {best_val_model_parameters[0]}, width {best_val_model_parameters[1]}")
    plot_model_loss_acc(train_losses_worst, val_losses_worst, test_losses_worst,
                        train_accs_worst, val_accs_worst, test_accs_worst,
                        losses_title=f"Losses over Worst Model: depth {worst_val_model_parameters[0]}, width {worst_val_model_parameters[1]}",
                        accuracies_title=f"Accuracies over Worst Model: depth {worst_val_model_parameters[0]}, width {worst_val_model_parameters[1]}")

    test_data = pd.read_csv('test.csv')
    plot_decision_boundaries(best_val_model, test_data[['long', 'lat']].values, test_data['country'].values,
                             f'Best model: depth {best_val_model_parameters[0]}, width {best_val_model_parameters[1]} Decision Boundaries', implicit_repr=False)
    plot_decision_boundaries(worst_val_model, test_data[['long', 'lat']].values, test_data['country'].values,
                             f'Worst model: depth {worst_val_model_parameters[0]}, width {worst_val_model_parameters[1]} Decision Boundaries', implicit_repr=False)

def test_depth_or_width_of_network(output_dimension, models_parameters, x_axis_parameters, parameter, title, x_label_title):
    final_results = {key: [0, 0, 0] for key in x_axis_parameters}

    for depth, width in models_parameters:
        print(f"------------------------- (Depth: {depth}, width: {width})-------------------------------")
        model = MLP(depth, width, output_dimension)
        _, train_accs, val_accs, test_accs, train_losses, val_losses, test_losses, _, _ = (
            train(train_dataset, val_dataset,test_dataset, model))
        if parameter == DEPTH:
            final_results[depth][0] = train_accs[-1]
            final_results[depth][1] = val_accs[-1]
            final_results[depth][2] = test_accs[-1]
        else:
            final_results[width][0] = train_accs[-1]
            final_results[width][1] = val_accs[-1]
            final_results[width][2] = test_accs[-1]
    train_accuracies =  [value[0] for value in final_results.values()]
    validation_accuracies = [value[1] for value in final_results.values()]
    test_accuracies = [value[2] for value in final_results.values()]
    x_axis = list(final_results.keys())
    plt.figure()
    plt.plot(x_axis, train_accuracies, label='Training')
    plt.plot(x_axis, validation_accuracies, label='Validation')
    plt.plot(x_axis, test_accuracies, label='Test')
    plt.title(title)
    plt.xlabel(x_label_title)
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

def plot_gradient_magnitudes(output_dimension):
    model = MLP(100, 4, output_dimension, extra_batch_norm=False, resnet=False, dropout=False)
    _, _, _, _, _, _, _, _, grad_magnitudes = train(train_dataset, val_dataset, test_dataset, model,
                                                    clipping=False, lr=0.001, batch_size=256, epochs=50, schedule=False)

    model = MLP(100, 40, output_dimension,extra_batch_norm=False, resnet=True, dropout=False)
    _, train_accs, val_accs, test_accs, train_losses, val_losses, test_losses, _, grad_magnitudes = train(train_dataset, val_dataset,test_dataset, model,
                                                   clipping=True, lr= 0.0005, batch_size=512, epochs=50, schedule=True,
                                                   regularization=True)
    plot_model_loss_acc(val_losses, test_losses, train_losses, train_accs, val_accs, test_accs,is_accuracy=True)
    plt.figure()
    for layer_idx, magnitudes in grad_magnitudes.items():
        plt.plot(magnitudes, label=f'Layer {layer_idx}')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Gradient Magnitude')
    plt.title('Mean Gradient Magnitudes for Selected Layers')
    plt.legend()
    plt.show()


class SinusoidalPreprocessor:
    def __init__(self, num_functions=10, alpha_start=0.1, alpha_end=1.0):
        """
        Initializes the sinusoidal preprocessor.
        Args:
            num_functions (int): Number of sine functions to apply.
            alpha_start (float): Starting value of alpha.
            alpha_end (float): Ending value of alpha.
        """
        self.alphas = torch.linspace(alpha_start, alpha_end, num_functions)

    def transform(self, features):
        """
        Transforms the input features by applying sine functions.
        Args:
            features (torch.Tensor): Input features of shape (N, D).
        Returns:
            torch.Tensor: Transformed features of shape (N, D * num_functions).
        """
        transformed_features = []
        for alpha in self.alphas:
            transformed_features.append(torch.sin(alpha * features))
        return torch.cat(transformed_features, dim=1)

def compare_sine_model_to_default(output_dimension):
    for i in range(2):
        model_type = "Sined model" if i == 0 else "Standard model"
        input_dim = 20 if i == 0 else 2
        print(f"-------------------------{model_type}-------------------------------\n")

        model = MLP(6, 16, output_dimension, input_dim = input_dim, is_sine=True if i == 0 else False)
        _, train_accs, val_accs, test_accs, train_losses, val_losses, test_losses, _, _ =(
            train(train_dataset, val_dataset,test_dataset, model, to_sine_data=True if i == 0 else False))
        plot_model_loss_acc(val_losses, test_losses, train_losses, train_accs, val_accs, test_accs,
                            is_accuracy=True if i == 0 else False)
        test_data = pd.read_csv('test.csv')
        plot_decision_boundaries(model, test_data[['long', 'lat']].values, test_data['country'].values,
                             f" {model_type} Decision Boundaries",
                             implicit_repr=True if i == 0 else False)


if __name__ == '__main__':
    # seed for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)
    train_dataset = EuropeDataset('train.csv')
    val_dataset = EuropeDataset('validation.csv')
    test_dataset = EuropeDataset('test.csv')

    #### YOUR CODE HERE #####
    output_dim = len(train_dataset.labels.unique())

    test_model_arbitrary_lr(output_dim, train_dataset, val_dataset, test_dataset, losses_title="Losses regular model")
    test_learning_rates_losses(output_dim)
    test_model_arbitrary_lr(output_dim, train_dataset, val_dataset, test_dataset, epochs=100)
    test_model_arbitrary_lr(output_dim, train_dataset, val_dataset, test_dataset, losses_title="Losses modified model", extra_batch_norm=True)
    test_validation_accuracies_per_batch_epoch(output_dim)
    test_train_losses_per_batch_epoch(output_dim)
    evaluate_MLP_performance(output_dim)
    test_depth_or_width_of_network(output_dim,
                     models_parameters=MODELS_WITH_WIDTH_16,
                     x_axis_parameters=HIDDEN_LAYERS_WIDTH_16,
                     parameter=DEPTH,
                     title='Accuracy vs Number of hidden layers', x_label_title='Hidden Layers')
    test_depth_or_width_of_network(output_dim,
                    models_parameters=MODELS_WITH_DEPTH_6,
                    x_axis_parameters=NUMER_NEURONS_DEPTH_6,
                    parameter=WIDTH,
                    title='Accuracy vs Number of neurons', x_label_title='Neurons')
    plot_gradient_magnitudes(output_dim)
    compare_sine_model_to_default(output_dim)
