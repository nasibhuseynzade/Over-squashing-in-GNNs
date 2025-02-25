# R2 Score Across Different depths for GAT Model on QM9

import os, sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CURRENT_DIR))

import torch
import matplotlib.pyplot as plt
import pickle

from preprocessing.fosr import edge_rewire
from models.training import _train_model_new
from models.models import GATModel, GINModel

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

try:
    with open('/home/huseynzade/SRP/datasets/qm9_dataset.pkl', 'rb') as f:
        dataset = pickle.load(f)
except FileNotFoundError:
    with open('/Users/nasibhuseynzade/Downloads/qm9_dataset.pkl','rb') as f:
        dataset = pickle.load(f)

# Define the depths and number of epochs
hidden_dims = [64,128,256]
num_epochs = 2

# First plot: GAT Model
plt.figure(figsize=(10, 6))

# Plot for each depth on the same axes for GAT
for hidden_dim in hidden_dims:
    print(f"\nTraining GAT model with hidden dimmension {hidden_dim}\n" + "-"*80)
    gat_model = GATModel(num_features=dataset.num_features, num_classes=1, hidden_dim=hidden_dim).to(device)
    gat_r2_scores = _train_model_new(gat_model, dataset, batch_size=64, learning_rate=0.0005, target_idx=0, num_epochs=num_epochs)
    
    # Plot the R2 scores with a different color for each depth
    plt.plot(range(num_epochs), gat_r2_scores, label=f'Hidden dimension = {hidden_dim}')

# Add title and labels for GAT plot
plt.title("R2 Score Across Different Number of Layers for GAT Model", fontsize=16)
plt.xlabel('Epochs')
plt.ylabel('R2 Score')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# Make the plot look nice
plt.tight_layout()

# Save the GAT plot to the results folder
results_folder = '/Users/nasibhuseynzade/Desktop/SRP_code/results' if os.path.exists('/Users/nasibhuseynzade/Desktop/SRP_code/results') else '/home/huseynzade/SRP/results'
gat_plot_path = os.path.join(results_folder, 'GAT_hiddendim_24feb.png')
plt.savefig(gat_plot_path, format='png')
plt.close()  # Close the current figure before creating the next one

# Second plot: GIN Model
plt.figure(figsize=(10, 6))

# Plot for each depth on the same axes for GIN
for hidden_dim in hidden_dims:
    print(f"\nTraining GIN model with hidden dimension {hidden_dim}\n" + "-"*80)
    gin_model = GINModel(num_features=dataset.num_features, num_classes=1, hidden_dim=hidden_dim).to(device)
    gin_r2_scores = _train_model_new(gin_model, dataset, batch_size=64, learning_rate=0.0005, target_idx=0, num_epochs=num_epochs)
    
    # Plot the R2 scores with a different color for each depth
    plt.plot(range(num_epochs), gin_r2_scores, label=f'Hidden dimmension = {hidden_dim}')

# Add title and labels for GIN plot
plt.title("R2 Score Across Different Number of Layers for GIN Model", fontsize=16)
plt.xlabel('Epochs')
plt.ylabel('R2 Score')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# Make the plot look nice
plt.tight_layout()

# Save the GIN plot to the results folder
gin_plot_path = os.path.join(results_folder, 'GIN_hiddendim_24feb.png')
plt.savefig(gin_plot_path, format='png')
plt.show()  # Show the second plot (optional)