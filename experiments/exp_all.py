# R2 Score Across Different depths for GAT Model on QM9

import os, sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CURRENT_DIR))

import torch
import matplotlib.pyplot as plt
import pickle

from preprocessing.sdrf import sdrf
from preprocessing.fosr import edge_rewire
from models.training import train_model, _train_model
from models.models import GATModel, GINModel

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


with open('/Users/nasibhuseynzade/Downloads/qm9_dataset.pkl','rb') as f:
   dataset = pickle.load(f)

sdrf_dataset=dataset.copy()
fosr_dataset=dataset.copy()

for i in range(len(dataset)):
    
    edge_index, edge_type, _ = edge_rewire(dataset[i].edge_index.numpy(), num_iterations=2)
    fosr_dataset[i].edge_index = torch.tensor(edge_index)
    fosr_dataset[i].edge_type = torch.tensor(edge_type)


for i in range(len(dataset)):
    sdrf_dataset[i].edge_index, sdrf_dataset[i].edge_type = sdrf(dataset[i], loops=2, remove_edges=False, is_undirected=True)


# Run experiments and plot results for depths 3, 4, and 5
depths = [3, 4, 5]
num_epochs = 2

# Create figures and subplots for GIN and GAT models
fig, axs = plt.subplots(1, 3, figsize=(18, 6))
fig2, axs2 = plt.subplots(1, 3, figsize=(18, 6))

fig.suptitle("R2 Score Across Different Number of Layers for GIN Model", fontsize=16)
fig2.suptitle("R2 Score Across Different Number of Layers for GAT Model", fontsize=16)

for i, depth in enumerate(depths):
    print(f"\nTraining GIN model with depth {depth}\n" + "-"*80)
    gin_model = GINModel(num_features=dataset.num_features, num_classes=1, depth=depth).to(device)
    gat_model = GATModel(num_features=dataset.num_features, num_classes=1, depth=depth).to(device)
    
    original_r2_scores_gin, fosr_r2_scores_gin, sdrf_r2_scores_gin = train_model(gin_model, dataset, fosr_dataset, sdrf_dataset, batch_size=64, learning_rate=0.0005, target_idx=0, num_epochs=num_epochs)
    print(f"\nTraining GAT model with depth {depth}\n" + "-"*80)
    original_r2_scores_gat, fosr_r2_scores_gat, sdrf_r2_scores_gat = train_model(gat_model, dataset, fosr_dataset, sdrf_dataset, batch_size=64, learning_rate=0.0005, target_idx=0, num_epochs=num_epochs)
     # Plot GIN model results
    axs[i].plot(range(num_epochs), original_r2_scores_gin, label='Original R2', color='blue')
    axs[i].plot(range(num_epochs), sdrf_r2_scores_gin, label='SDRF R2', color='orange')
    axs[i].plot(range(num_epochs), fosr_r2_scores_gin, label='FOSR R2', color='green')
    axs[i].set_title(f'Depth = {depth}')
    axs[i].set_xlabel('Epochs')
    axs[i].set_ylabel('R2 Score')
    axs[i].legend()

    # Plot GAT model results
    axs2[i].plot(range(num_epochs), original_r2_scores_gat, label='Original R2', color='blue')
    axs2[i].plot(range(num_epochs), sdrf_r2_scores_gat, label='SDRF R2', color='orange')
    axs2[i].plot(range(num_epochs), fosr_r2_scores_gat, label='FOSR R2', color='green')
    axs2[i].set_title(f'Depth = {depth}')
    axs2[i].set_xlabel('Epochs')
    axs2[i].set_ylabel('R2 Score')
    axs2[i].legend()

plt.tight_layout()

# Results folder
results_folder = '/Users/nasibhuseynzade/Desktop/SRP_code/results'

# Save the figures separately
gin_plot_path = os.path.join(results_folder, 'GIN_Depth_R2_Comparison.png')
gat_plot_path = os.path.join(results_folder, 'GAT_Depth_R2_Comparison.png')

fig.savefig(gin_plot_path, format='png')
fig2.savefig(gat_plot_path, format='png')

# Close the figures to free up memory
plt.close(fig)
plt.close(fig2)