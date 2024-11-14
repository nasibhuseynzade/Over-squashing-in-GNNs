import torch
from torch_geometric.datasets import QM9
from torch_geometric.utils import to_networkx, to_undirected
import networkx as nx
import numpy as np
import torch.nn as nn
from scipy.linalg import pinv
from math import inf
from numba import jit, int64
from tqdm import tqdm
from torch_geometric.utils import to_scipy_sparse_matrix
import torch.nn.functional as F
from torch.nn import ModuleList, Dropout, ReLU
from torch_geometric.nn import GCNConv, GATConv, GINConv, global_mean_pool
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import NormalizeFeatures
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.data import DataLoader as GeoDataLoader
import networkx as nx
import numpy as np
from scipy.linalg import pinv
import pandas as pd
import matplotlib.pyplot as plt

import os, sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CURRENT_DIR))

from preprocessing.fosr import edge_rewire

dataset = QM9(root='qm9_data')

def compute_commute_time(graph):
    """
    Compute the commute time for each pair of nodes in a graph.
    :param graph: A NetworkX graph object.
    :return: Commute time matrix (numpy array).
    """
    # Convert graph to adjacency matrix (scipy sparse format)
    adj_matrix = nx.adjacency_matrix(graph)

    # Compute degree matrix
    degrees = np.array(adj_matrix.sum(axis=1)).flatten()
    D = np.diag(degrees)

    # Compute Laplacian matrix L = D - A
    L = D - adj_matrix.toarray()

    # Compute the pseudoinverse of the Laplacian
    L_pseudo = pinv(L)

    # Compute commute time between all pairs of nodes
    num_nodes = L.shape[0]
    commute_time = np.zeros((num_nodes, num_nodes))

    for i in range(num_nodes):
        for j in range(num_nodes):
            commute_time[i, j] = L_pseudo[i, i] + L_pseudo[j, j] - 2 * L_pseudo[i, j]

    return commute_time


def aggregate_commute_times(graph):
    """
    Aggregate the commute times for a single graph.
    :param graph: A NetworkX graph object.
    :return: Average commute time across the graph.
    """
    commute_times = compute_commute_time(graph)

    # Get upper triangular part of the commute time matrix (since it's symmetric)
    upper_triangle_indices = np.triu_indices_from(commute_times, k=1)
    commute_times_upper = commute_times[upper_triangle_indices]

    # Return average commute time
    return np.mean(commute_times_upper)


def compute_average_commute_time_rewired_qm9_loops(dataset):

    
    # Generate loop values from 2 to 50 in steps of 2
    loop_values = list(range(1, 2, 1))
    results = []
    
    for loops in tqdm(loop_values, desc="Processing loop values"):
        #original_commute_times = []
        rewired_commute_times = []
        
        for data in dataset:
            # Convert edge_index to torch tensor if it's numpy array
            if isinstance(data.edge_index, np.ndarray):
                data.edge_index = torch.from_numpy(data.edge_index)
            
            # Original graph commute time
            #original_graph = to_networkx(data, node_attrs=[], edge_attrs=[], to_undirected=True)
            #original_ct = aggregate_commute_times(original_graph)
            #original_commute_times.append(original_ct)
            
            # Rewire the graph using SDRF
            rewired_edge_index, edge_type, _ = edge_rewire(data.edge_index.numpy(), num_iterations=loops)
            
            # Create new data object with rewired edges
            rewired_data = data.clone()
            # Convert rewired edge_index to torch tensor if it's numpy array
            if isinstance(rewired_edge_index, np.ndarray):
                rewired_edge_index = torch.from_numpy(rewired_edge_index)
            rewired_data.edge_index = rewired_edge_index
            
            # Calculate commute time for rewired graph
            rewired_graph = to_networkx(rewired_data, node_attrs=[], edge_attrs=[], to_undirected=True)
            rewired_ct = aggregate_commute_times(rewired_graph)
            rewired_commute_times.append(rewired_ct)
        
        # Calculate averages
        #orig_avg = np.mean(original_commute_times)
        rewired_avg = np.mean(rewired_commute_times)
        print(f"\nResults for {loops} loops:")
        #print(f"Original average commute time: {orig_avg:.4f}")
        print(f"Rewired average commute time: {rewired_avg:.4f}")
        
        results.append({
            'loops': loops,
            #'originalAvg': orig_avg,
            'rewiredAvg': rewired_avg
        })
    
    return results

# Usage:
results = compute_average_commute_time_rewired_qm9_loops(dataset)



plot_data = pd.DataFrame(results)

plt.figure(figsize=(10, 6))
#plt.plot(plot_data['loops'], plot_data['originalAvg'], 'o-', label='Original Average')
plt.plot(plot_data['loops'], plot_data['rewiredAvg'], 'o-', label='Rewired Average')
plt.xlabel('Number of Loops')
plt.ylabel('Average Commute Time')
plt.title('Commute Time Analysis by Number of Loops')
plt.legend()
plt.grid(True)
plt.show()

