import os, sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CURRENT_DIR))

import torch
from torch_geometric.datasets import QM9
from torch_geometric.transforms import NormalizeFeatures
import matplotlib.pyplot as plt
import pickle

from preprocessing.fosr import edge_rewire
from models.training import train_model
from models.models import GINModel

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('/Users/nasibhuseynzade/Downloads/qm9_dataset.pkl','rb') as f:
   dataset = pickle.load(f)

fosr_dataset=dataset.copy()

print("Rewiring started")
for i in range(len(dataset)):
    
    edge_index, edge_type, _ = edge_rewire(dataset[i].edge_index.numpy(), num_iterations=2)
    fosr_dataset[i].edge_index = torch.tensor(edge_index)
    fosr_dataset[i].edge_type = torch.tensor(edge_type)
print("Rewiring ended")

# Run experiments and plot results for depths 3, 4, and 5
depths = [3, 4, 5]
num_epochs = 3

fig, axs = plt.subplots(1, 3, figsize=(18, 6))

for i, depth in enumerate(depths):
    print(f"\nTraining GIN model with depth {depth}\n" + "-"*80)
    model=GINModel(num_features=dataset.num_features, num_classes=1, depth=depth).to(device)
    train_losses, original_r2_scores, rewired_r2_scores = train_model(model, dataset, fosr_dataset, target_idx=0, num_epochs=num_epochs)
    
    axs[i].plot(range(num_epochs), original_r2_scores, label='Original Dataset R2', color='blue')
    axs[i].plot(range(num_epochs), rewired_r2_scores, label='Rewired Dataset R2', color='orange')
    axs[i].set_title(f'Depth = {depth}')
    axs[i].set_xlabel('Epochs')
    axs[i].set_ylabel('R2 Score')
    axs[i].legend()

plt.tight_layout()
