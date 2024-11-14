import torch
import numpy as np
from torch_geometric.utils import to_scipy_sparse_matrix
from scipy.sparse import csgraph
from scipy.linalg import eigh


def compute_spectral_gap(data):
    # Convert edge index to scipy sparse matrix (adjacency matrix)
    adj = to_scipy_sparse_matrix(data.edge_index, num_nodes=data.num_nodes)
    
    # Compute the Laplacian matrix from the adjacency matrix
    laplacian = csgraph.laplacian(adj, normed=False)
    
    # Compute the eigenvalues of the Laplacian matrix
    eigenvalues = eigh(laplacian.toarray(), eigvals_only=True)
    
    # Spectral gap is the difference between the second smallest and the first eigenvalue (0)
    spectral_gap = eigenvalues[1]  # The second smallest eigenvalue
    
    return spectral_gap
