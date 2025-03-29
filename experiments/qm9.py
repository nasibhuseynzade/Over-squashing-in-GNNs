# R2 Score Across Different depths for GAT and GIN Models on QM9

import os, sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CURRENT_DIR))

import torch
import matplotlib.pyplot as plt
import pickle

from preprocessing.ct_sdrf import sdrf_with_ct_threshold
from preprocessing.sdrf import sdrf
from models_and_training.training import train_model
from models_and_training.models import GINModel

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


try:
    with open('/home/huseynzade/SRP/datasets/qm9_dataset.pkl', 'rb') as f:
        dataset = pickle.load(f)
except FileNotFoundError:
    with open('/Users/nasibhuseynzade/Downloads/qm9_dataset.pkl','rb') as f:
        dataset = pickle.load(f)

sdrf_dataset=dataset.copy()
ct_dataset=dataset.copy()

for i in range(len(dataset)):
    ct_dataset[i].edge_index, ct_dataset[i].edge_type = sdrf_with_ct_threshold(dataset[i], ct_threshold=2.5, remove_edges=False, is_undirected=True)


for i in range(len(dataset)):
    sdrf_dataset[i].edge_index, sdrf_dataset[i].edge_type = sdrf(dataset[i], loops=50, remove_edges=False, is_undirected=True)



num_epochs = 100

# Create a single figure
fig = plt.figure(figsize=(10, 6))

plt.title("R2 Score for GIN Model", fontsize=16)

gin_model = GINModel(num_features=dataset.num_features, num_classes=1, depth=4, hidden_dim=128).to(device)
        
original_r2_scores_gin, sdrf_r2_scores_gin, ct_r2_scores_gin = train_model(gin_model, dataset, ct_dataset, sdrf_dataset, num_epochs=num_epochs, batch_size=64, learning_rate=0.0005)


# Plot GIN model results
plt.plot(range(num_epochs), original_r2_scores_gin, label='Original R2', color='blue')
plt.plot(range(num_epochs), sdrf_r2_scores_gin, label='SDRF R2', color='orange')
plt.plot(range(num_epochs), ct_r2_scores_gin, label='CT_SDRF R2', color='green')
plt.xlabel('Epochs')
plt.ylabel('R2 Score')
plt.legend()

plt.tight_layout()

# Results folder
results_folder = '/Users/nasibhuseynzade/Desktop/SRP_code/results' if os.path.exists('/Users/nasibhuseynzade/Desktop/SRP_code/results') else '/home/huseynzade/SRP/results'

# Save the figure
gin_plot_path = os.path.join(results_folder, 'qm_all.png')
fig.savefig(gin_plot_path, format='png')

# Close the figure to free up memory
plt.close(fig)