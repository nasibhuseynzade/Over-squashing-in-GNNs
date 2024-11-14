from preprocessing.fosr import edge_rewire
import numpy as np

def apply_fosr(edge_index, rewire_fraction=0.05, min_iterations=10, max_iterations=1000):
    edge_type = np.zeros(edge_index.shape[1], dtype=np.int64)
    n = np.max(edge_index) + 1
    x = 2 * np.random.random(n) - 1

    # Calculate num_iterations based on the number of edges and rewire_fraction
    num_edges = edge_index.shape[1]
    num_iterations = int(num_edges * rewire_fraction)

    # Ensure num_iterations is within a reasonable range
    num_iterations = max(min_iterations, min(num_iterations, max_iterations))

    #print(f"Number of iterations for FOSR: {num_iterations}")

    new_edge_index, new_edge_type, _ = edge_rewire(
        edge_index,
        x=x,
        edge_type=edge_type,
        num_iterations=num_iterations,
        initial_power_iters=5
    )

    return new_edge_index, new_edge_type