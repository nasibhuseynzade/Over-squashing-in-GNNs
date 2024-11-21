import os, sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CURRENT_DIR))

from numba import jit, prange
import numpy as np
import torch
from torch_geometric.utils import to_networkx, from_networkx


from preprocessing.sdrf import balanced_forman_post_delta, balanced_forman_curvature
from metrics.commute_time import avg_commute_times

def hybrid_rewiring(
    data,
    loops=10,
    remove_edges=True,
    removal_bound=0.5,
    tau=1,
    is_undirected=False,
    commute_threshold=0.1  # threshold for accepting commute time changes
):
    N = data.x.shape[0]
    A = np.zeros(shape=(N, N))
    m = data.edge_index.shape[1]

    if not "edge_type" in data.keys:
        edge_type = np.zeros(m, dtype=int)
    else:
        edge_type = data.edge_type

    # Build initial adjacency matrix
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
    C = np.zeros((N, N))
    
    initial_commute_time = avg_commute_times(G)
    current_commute_time = initial_commute_time

    for x in range(loops):
        can_add = True
        # Step 1: Use original mechanism to propose changes
        balanced_forman_curvature(A, C=C)
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
                    candidates.append((i, j))

        if len(candidates):
            # Step 2: Calculate improvements using original method
            D = balanced_forman_post_delta(A, x, y, x_neighbors, y_neighbors)
            improvements = []
            for (i, j) in candidates:
                improvements.append(
                    (D - C[x, y])[x_neighbors.index(i), y_neighbors.index(j)]
                )

            # Step 3: Get top K candidates based on original criteria
            K = min(3, len(candidates))  # Consider top 3 candidates
            top_k_indices = np.argsort(improvements)[-K:]
            best_edge = None
            best_commute_time = current_commute_time

            # Step 4: Evaluate these candidates using commute time
            for idx in top_k_indices:
                k, l = candidates[idx]
                # Temporarily add edge
                G.add_edge(k, l)
                new_commute_time = avg_commute_times(G)
                G.remove_edge(k, l)

                # If commute time improves or doesn't worsen significantly
                if new_commute_time < best_commute_time:
                    best_commute_time = new_commute_time
                    best_edge = (k, l)

            # Step 5: Add the best edge if found
            if best_edge is not None:
                k, l = best_edge
                G.add_edge(k, l)
                edge_type = np.append(edge_type, 1)
                edge_type = np.append(edge_type, 1)
                if is_undirected:
                    A[k, l] = A[l, k] = 1
                else:
                    A[k, l] = 1
                current_commute_time = best_commute_time
            else:
                can_add = False
                if not remove_edges:
                    break

        # Step 6: Handle edge removal
        if remove_edges:
            ix_max = C.argmax()
            x = ix_max // N
            y = ix_max % N
            if C[x, y] > removal_bound:
                # Check commute time before removing
                G.remove_edge(x, y)
                new_commute_time = avg_commute_times(G)
                
                # Only remove if it doesn't worsen commute time significantly
                if new_commute_time - current_commute_time < commute_threshold:
                    if is_undirected:
                        A[x, y] = A[y, x] = 0
                    else:
                        A[x, y] = 0
                    current_commute_time = new_commute_time
                else:
                    # Restore edge if removal worsens commute time too much
                    G.add_edge(x, y)
            else:
                if can_add is False:
                    break

    return from_networkx(G).edge_index, torch.tensor(edge_type)