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
from torch_geometric.nn import GINConv, global_add_pool, BatchNorm
import matplotlib.pyplot as plt
from torch_geometric.data import DataLoader as GeoDataLoader

class GINModel(torch.nn.Module):
    def __init__(self, num_features, num_classes=1, hidden_dim=64, depth=3):
        super(GINModel, self).__init__()

        # Define GIN layers
        self.convs = torch.nn.ModuleList()
        self.convs.append(GINConv(
            torch.nn.Sequential(
                torch.nn.Linear(num_features, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU()
            )
        ))

        # Additional GIN layers
        for _ in range(depth - 1):
            self.convs.append(GINConv(
                torch.nn.Sequential(
                    torch.nn.Linear(hidden_dim, hidden_dim),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_dim, hidden_dim),
                    torch.nn.ReLU()
                )
            ))

        # Batch normalization layers
        self.batch_norms = torch.nn.ModuleList([BatchNorm(hidden_dim) for _ in range(depth)])

        # Final regression layer
        self.final_lin = torch.nn.Linear(hidden_dim, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = conv(x, edge_index)
            x = batch_norm(x)
            x = F.relu(x)
        x = global_add_pool(x, data.batch)  # Pool to get a graph-level representation
        x = self.final_lin(x)  # Final regression output
        return x


class GATModel(nn.Module):
    def __init__(self, num_features, num_classes, hidden_dim=64, depth=3):
        super(GATModel, self).__init__()
        self.layers = nn.ModuleList()

        # First GAT layer
        self.layers.append(GATConv(num_features, hidden_dim, heads=4, concat=True))

        # Hidden GAT layers based on depth
        for _ in range(1, depth - 1):
            self.layers.append(GATConv(hidden_dim * 4, hidden_dim, heads=4, concat=True))

        # Final GAT layer
        self.layers.append(GATConv(hidden_dim * 4, hidden_dim, heads=4, concat=True))

        # Fully-connected layer for output
        self.fc = nn.Linear(hidden_dim * 4, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Apply each GAT layer
        for conv in self.layers:
            x = conv(x, edge_index)

        # Global pooling over nodes in each graph in the batch
        x = global_add_pool(x, batch)

        # Final classification layer
        x = self.fc(x)

        return x