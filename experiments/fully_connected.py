import os, sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CURRENT_DIR))

import torch
from torch_geometric.data import Data
import pickle
import matplotlib.pyplot as plt

from models_and_training.models import GINModel
from models_and_training.training import train_model


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def make_fully_connected(data):

    num_nodes = data.x.size(0)

    # Create all possible pairs of nodes
    rows = []
    cols = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:  # Exclude self-loops
                rows.append(i)
                cols.append(j)

    # Create new edge_index
    new_edge_index = torch.tensor([rows, cols], dtype=torch.long)

    # Create new graph with fully connected edges
    new_data = Data(
        x=data.x,
        edge_index=new_edge_index,
        y=data.y if hasattr(data, 'y') else None,
        pos=data.pos if hasattr(data, 'pos') else None
    )

    return new_data

# Function to process entire dataset
def convert_dataset_to_fully_connected(dataset):

    #Convert entire QM9 dataset to fully connected graphs

    converted_dataset = []
    for data in dataset:
        converted_data = make_fully_connected(data)
        converted_dataset.append(converted_data)

    return converted_dataset


try:
    with open('/home/huseynzade/SRP/datasets/qm9_dataset.pkl', 'rb') as f:
        dataset = pickle.load(f)
except FileNotFoundError:
    with open('/Users/nasibhuseynzade/Downloads/qm9_dataset.pkl','rb') as f:
        dataset = pickle.load(f)

# Convert to fully connected
fully_connected_dataset = convert_dataset_to_fully_connected(dataset)

num_epochs = 100

model=GINModel(num_features=dataset.num_features, num_classes=1, depth=4, hidden_dim=128).to(device)

train_losses, original_r2_scores, fully_connected_r2_scores = train_model(model, dataset, fully_connected_dataset, batch_size=64, learning_rate=0.0005, target_idx=0, num_epochs=num_epochs)

# Assuming `num_epochs` is the number of epochs
epochs = range(1, num_epochs + 1)

# Plot the R² values for both the original and fully connected graphs
plt.figure(figsize=(10, 6))
plt.plot(epochs, original_r2_scores, label='Original Graph R²', color='blue', linestyle='-', marker='o')
plt.plot(epochs, fully_connected_r2_scores, label='Fully Connected Graph R²', color='red', linestyle='-', marker='o')

# Add labels, title, and legend
plt.xlabel('Epochs')
plt.ylabel('R² Score')
plt.title('R² Scores of Original and Fully Connected Graphs Over Epochs')
plt.legend()
plt.grid(True)

# Define the folder for results
results_folder = '/home/huseynzade/SRP/results'
os.makedirs(results_folder, exist_ok=True)  # Create the folder if it doesn't exist

# Save the plot to the results folder
plot_path = os.path.join(results_folder, 'fully_connected.png')
plt.savefig(plot_path, format='png')

# Show the plot
plt.show()

