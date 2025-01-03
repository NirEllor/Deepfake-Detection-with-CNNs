import torch
import torch.nn as nn
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from helpers import *
import pandas as pd

class EuropeDataset(Dataset):
    def __init__(self, csv_file):
        """
        Args:
            csv_file (string): Path to the CSV file with annotations.
        """
        #### YOUR CODE HERE ####
        # Load the data into a tensors
        # The features shape is (n,d)
        # The labels shape is (n)
        # The feature dtype is float
        # THe labels dtype is long

        #### END OF YOUR CODE ####
        pass

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        #### YOUR CODE HERE ####
        pass

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the data row
        
        Returns:
            dictionary or list corresponding to a feature tensor and it's corresponding label tensor
        """
        #### YOUR CODE HERE ####
        pass
    

class MLP(nn.Module):
    def __init__(self, num_hidden_layers, hidden_dim, output_dim):
        super(MLP, self).__init__()
        """
        Args:
            num_hidden_layers (int): The number of hidden layers, in total you'll have an extra layer at the end, from hidden_dim to output_dim
            hidden_dim (int): The hidden layer dimension
            output_dim (int): The output dimension, should match the number of classes in the dataset
        """
        #### YOUR CODE HERE ####
        pass

    def forward(self, x):
        #### YOUR CODE HERE ####
       pass


def train(train_dataset, val_dataset, test_dataset, model, lr=0.001, epochs=50, batch_size=256):    

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    valloader = torch.utils.data.DataLoader(val_dataset, batch_size=1024, shuffle=False, num_workers=0)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=1024, shuffle=False, num_workers=0)    
    
    #### YOUR CODE HERE ####
    # initialize your criterion and optimizer here
    # criterion = 
    # optimizer = 

    train_accs = []
    val_accs = []
    test_accs = []
    train_losses = []
    val_losses = []
    test_losses = []

    for ep in range(epochs):
        #
        model.train()
        #### YOUR CODE HERE ####
        # perform training epoch here


        model.eval()
        with torch.no_grad():
            #### YOUR CODE HERE ####
            # perform validation loop and test loop here
            pass

        print('Epoch {:}, Train Acc: {:.3f}, Val Acc: {:.3f}, Test Acc: {:.3f}'.format(ep, train_accs[-1], val_accs[-1], test_accs[-1]))        

    model, train_accs, val_accs, test_accs, train_losses, val_losses, test_losses 



if __name__ == '__main__':
    # seed for reproducibility
    torch.manual_seed(0)    

    train_dataset = EuropeDataset('train.csv')
    val_dataset = EuropeDataset('validation.csv')
    test_dataset = EuropeDataset('test.csv')

    #### YOUR CODE HERE #####
    # Find the number of classes, e.g.:
    # output_dim = len(train_dataset.labels.unique()) 
    model = MLP(6, 16, output_dim)
    


    model, train_accs, val_accs, test_accs, train_losses, val_losses, test_losses = train(train_dataset, val_dataset, test_dataset, model, lr=0.001, epochs=50, batch_size=256)

    plt.figure()
    plt.plot(train_losses, label='Train', color='red')
    plt.plot(val_losses, label='Val', color='blue')
    plt.plot(test_losses, label='Test', color='green')
    plt.title('Losses')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(train_accs, label='Train', color='red')
    plt.plot(val_accs, label='Val', color='blue')
    plt.plot(test_accs, label='Test', color='green')
    plt.title('Accs.')
    plt.legend()
    plt.show()



    train_data = pd.read_csv('train.csv')
    val_data = pd.read_csv('validation.csv')
    test_data = pd.read_csv('test.csv')
    plot_decision_boundaries(model, test_data[['long', 'lat']].values, test_data['country'].values, 'Decision Boundaries', implicit_repr=False)