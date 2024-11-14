import os, sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CURRENT_DIR))

import torch
from torch_geometric.datasets import QM9
from torch_geometric.transforms import NormalizeFeatures
import matplotlib.pyplot as plt

from preprocessing.fosr import edge_rewire
from experiments.training import train_model
from models.models import GINModel

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = QM9(root='/tmp/QM9', transform=NormalizeFeatures())

fosr_dataset=dataset.copy()

print("Rewiring started")
for i in range(len(dataset)):
    
    edge_index, edge_type, _ = edge_rewire(dataset[i].edge_index.numpy(), num_iterations=1)
    fosr_dataset[i].edge_index = torch.tensor(edge_index)
    fosr_dataset[i].edge_type = torch.tensor(edge_type)
print("Rewiring ended")


hidden_dims = [64, 128, 256]
num_epochs = 50

fig, axs = plt.subplots(1, 3, figsize=(18, 6))

for i, hidden_dim in enumerate(hidden_dims):


    print(f"\nTraining GIN model with hidden dimension = {hidden_dim}\n" + "-"*80)
    model=GINModel(num_features=dataset.num_features, num_classes=1, hidden_dim=hidden_dim).to(device)
    train_losses, original_r2_scores, rewired_r2_scores = train_model(model, dataset, fosr_dataset, target_idx=0, hidden_dim=hidden_dim, num_epochs=num_epochs)

    axs[i].plot(range(num_epochs), original_r2_scores, label='Original Dataset R2', color='blue')
    axs[i].plot(range(num_epochs), rewired_r2_scores, label='Rewired Dataset R2', color='orange')
    axs[i].set_title(f'Hidden Dimension = {hidden_dim}')
    axs[i].set_xlabel('Epochs')
    axs[i].set_ylabel('R2 Score')
    axs[i].legend()

plt.tight_layout()
plt.show()