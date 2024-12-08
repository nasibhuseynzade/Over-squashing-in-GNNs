import torch
import numpy as np
from torch_geometric.utils import to_scipy_sparse_matrix
from scipy.sparse import csgraph
from scipy.linalg import eigh
import networkx as nx
from torch_geometric.utils import to_networkx

def calculate_spectral_gap(graph):
    """
    Calculate the spectral gap of a single graph.
    The spectral gap is the difference between the first and second smallest eigenvalues 
    of the normalized Laplacian matrix.
    """
    # Compute the normalized Laplacian matrix
    laplacian = nx.normalized_laplacian_matrix(graph).toarray()
    
    # Compute the eigenvalues of the Laplacian
    eigenvalues = np.linalg.eigvalsh(laplacian)
    
    # Sort eigenvalues in ascending order
    eigenvalues.sort()
    
    # The spectral gap is the difference between the 2nd and 1st eigenvalues
    spectral_gap = eigenvalues[1] - eigenvalues[0]
    return spectral_gap

def average_spectral_gap(dataset, num_graphs=None):
    """
    Calculate the average spectral gap for a dataset of molecular graphs.
    :param dataset: PyTorch Geometric dataset (e.g., QM9)
    :param num_graphs: Number of graphs to compute (default: all)
    """
    spectral_gaps = []
    num_graphs = num_graphs or len(dataset)
    
    for i in range(num_graphs):
        # Convert PyTorch Geometric data to NetworkX graph
        data = dataset[i]
        graph = to_networkx(data, to_undirected=True)
        
        # Calculate the spectral gap for the current graph
        spectral_gap = calculate_spectral_gap(graph)
        spectral_gaps.append(spectral_gap)
    
    return np.mean(spectral_gaps)