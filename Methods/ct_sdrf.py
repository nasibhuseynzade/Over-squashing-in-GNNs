# Our CT-SDRF method
import torch
from numba import prange
import numpy as np
from torch_geometric.utils import to_networkx, from_networkx
import os, sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CURRENT_DIR))
from metrics.commute_time import avg_commute_times

def _balanced_forman_curvature(A, A2, d_in, d_out, N, C):
    for i in prange(N):
        for j in prange(N):
            if A[i, j] == 0:
                C[i, j] = 0
                break

            if d_in[i] > d_out[j]:
                d_max = d_in[i]
                d_min = d_out[j]
            else:
                d_max = d_out[j]
                d_min = d_in[i]

            if d_max * d_min == 0:
                C[i, j] = 0
                break

            sharp_ij = 0
            lambda_ij = 0
            for k in range(N):
                TMP = A[k, j] * (A2[i, k] - A[i, k]) * A[i, j]
                if TMP > 0:
                    sharp_ij += 1
                    if TMP > lambda_ij:
                        lambda_ij = TMP

                TMP = A[i, k] * (A2[k, j] - A[k, j]) * A[i, j]
                if TMP > 0:
                    sharp_ij += 1
                    if TMP > lambda_ij:
                        lambda_ij = TMP

            C[i, j] = (
                (2 / d_max)
                + (2 / d_min)
                - 2
                + (2 / d_max + 1 / d_min) * A2[i, j] * A[i, j]
            )
            if lambda_ij > 0:
                C[i, j] += sharp_ij / (d_max * lambda_ij)


def balanced_forman_curvature(A, C=None):
    N = A.shape[0]
    A2 = np.matmul(A, A)
    d_in = A.sum(axis=0)
    d_out = A.sum(axis=1)
    if C is None:
        C = np.zeros((N, N))

    _balanced_forman_curvature(A, A2, d_in, d_out, N, C)
    return C

def best_ct_candidate(G, candidates):

    #Find the best candidate edge that minimizes the average commute time.

    # Calculate original commute time
    original_ct = avg_commute_times(G)
    
    best_edge = None
    min_ct = float('inf')
    
    # Try each candidate edge
    for edge in candidates:
        # Create a copy of the graph to avoid modifying the original
        G_temp = G.copy()
        
        # Add the candidate edge
        G_temp.add_edge(edge[0], edge[1])
        
        # Calculate new average commute time
        new_ct = avg_commute_times(G_temp)
        
        # Check if this edge provides a better (lower) commute time
        if new_ct < min_ct:
            min_ct = new_ct
            best_edge = edge
    
    # If we found an improvement, return the best edge
    if best_edge is not None and min_ct < original_ct:
        return best_edge
    
    # If no improvement was found (unlikely in most cases), return the first candidate
    return candidates[0] if candidates else None

def sdrf_with_ct_threshold(
    data,
    ct_threshold,  
    remove_edges=True,
    removal_bound=0.5,
    is_undirected=False,
    max_iterations=20  # Safety parameter to prevent infinite loops
):
    N = data.x.shape[0]
    A = np.zeros(shape=(N, N))
    m = data.edge_index.shape[1]
    
    if not "edge_type" in data.keys():
        edge_type = np.zeros(m, dtype=int)
    else:
        edge_type = data.edge_type
    
    if is_undirected:
        for i, j in zip(data.edge_index[0], data.edge_index[1]):
            if i != j:
                A[i, j] = A[j, i] = 1.0
    else:
        for i, j in zip(data.edge_index[0], data.edge_index[1]):
            if i != j:
                A[i, j] = 1.0
    
    N = A.shape[0]
    G = to_networkx(data)
    if is_undirected:
        G = G.to_undirected()
    
    # Calculate initial average commute time
    current_ct = avg_commute_times(G)
    
    iteration = 0
    
    # Continue until we reach the threshold or max iterations
    while current_ct > ct_threshold and iteration < max_iterations:
        iteration += 1

        
        C = balanced_forman_curvature(A)
        
        ix_min = C.argmin()
        x = ix_min // N
        y = ix_min % N
        
        if is_undirected:
            x_neighbors = list(G.neighbors(x)) + [x]
            y_neighbors = list(G.neighbors(y)) + [y]
        else:
            x_neighbors = list(G.successors(x)) + [x]
            y_neighbors = list(G.predecessors(y)) + [y]
        
        candidates = []
        
        for i in x_neighbors:
            for j in y_neighbors:
                if (i != j) and (not G.has_edge(i, j)):
                    if is_undirected:
                        if i < j:
                            candidates.append((i, j))
                        elif i > j and (j, i) not in candidates:
                            candidates.append((j, i))
                    else:
                        candidates.append((i, j))
        
        # Check if we have any candidates
        if not candidates:
            #print(f"No candidates found at iteration {iteration}. Exiting.")
            break
        
        # Find the best candidate edge
        k, l = best_ct_candidate(G, candidates)
        
        # Add the edge to our graph
        G.add_edge(k, l)
        edge_type = np.append(edge_type, 1)
        if is_undirected:
            edge_type = np.append(edge_type, 1)  # Only need to append twice for undirected
        
        # Update adjacency matrix
        if is_undirected:
            A[k, l] = A[l, k] = 1
        else:
            A[k, l] = 1
        
        # Recalculate commute time
        new_ct = avg_commute_times(G)
        
        # Check if we've improved
        if new_ct >= current_ct:

            if remove_edges:
                G.remove_edge(k, l)
                if is_undirected:
                    A[k, l] = A[l, k] = 0
                else:
                    A[k, l] = 0
        
        # Update current commute time
        current_ct = avg_commute_times(G)

    
    return from_networkx(G).edge_index, torch.tensor(edge_type)