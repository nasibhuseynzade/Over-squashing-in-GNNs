import torch
import pickle

import os, sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CURRENT_DIR))

from preprocessing.sdrf import sdrf
from preprocessing.fosr import edge_rewire
import matplotlib.pyplot as plt

with open('/Users/nasibhuseynzade/Downloads/qm9_dataset.pkl','rb') as f:
   dataset = pickle.load(f)


# Function to calculate average edges per graph
def calculate_average_edges(dataset):
    total_edges = 0
    num_graphs = len(dataset)
    for data in dataset:
        total_edges += data.edge_index.size(1)
    return total_edges / num_graphs
# Create lists to store average edges
fosr_avg_edges = []
sdrf_avg_edges = []

for num_iterations in range(1, 4):
    fosr_dataset = dataset.copy()
    sdrf_dataset = dataset.copy()
    fosr_edges = 0
    sdrf_edges = 0
    
    for i in range(len(dataset)):
        # FOSR rewiring
        edge_index_fosr, edge_type_fosr, _ = edge_rewire(dataset[i].edge_index.numpy(), num_iterations=num_iterations)
        fosr_dataset[i].edge_index = torch.tensor(edge_index_fosr)
        fosr_dataset[i].edge_type = torch.tensor(edge_type_fosr)
        fosr_edges += edge_index_fosr.shape[1]
        
        # SDRF rewiring
        edge_index_sdrf, edge_type_sdrf, _ = sdrf(dataset[i].edge_index.numpy(), num_iterations=num_iterations)
        sdrf_dataset[i].edge_index = torch.tensor(edge_index_sdrf)
        sdrf_dataset[i].edge_type = torch.tensor(edge_type_sdrf)
        sdrf_edges += edge_index_sdrf.shape[1]
    
    # Calculate and store average edges
    fosr_avg_edges.append(fosr_edges / len(dataset))
    sdrf_avg_edges.append(sdrf_edges / len(dataset))

# Create figure and plot
plt.figure(figsize=(10, 6))
plt.plot(range(1, 4), fosr_avg_edges, marker='o', label='FOSR')
plt.plot(range(1, 4), sdrf_avg_edges, marker='s', label='SDRF')
plt.xlabel('Number of Iterations')
plt.ylabel('Average Number of Edges per Graph')
plt.title('Average Number of Edges per Graph vs. Number of Iterations')
plt.legend()

# Save figure
plt.savefig('results/edges_comparison.png', dpi=300, bbox_inches='tight')
plt.show()