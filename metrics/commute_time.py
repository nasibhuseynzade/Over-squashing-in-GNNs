import networkx as nx
import numpy as np
from scipy.linalg import pinv

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


def avg_commute_times(graph):
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

def best_ct_candidate(G, candidates):
    """
    Find the best candidate edge that minimizes the average commute time.
    
    Parameters:
    G (networkx.Graph): The input graph
    candidates (list): List of tuples representing candidate edges to add, e.g., [(1,2), (1,3), ...]
    
    Returns:
    tuple: The best edge (k,l) that minimizes average commute time
    """
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
    # or you could implement some alternative strategy
    return candidates[0] if candidates else None
