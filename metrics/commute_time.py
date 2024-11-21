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

