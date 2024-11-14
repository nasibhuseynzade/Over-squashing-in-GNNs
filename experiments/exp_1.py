# First experiment
# Calculating commute time changes of original dataset and FoSR modified dataset

import os, sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CURRENT_DIR))
from metrics.commute_time import aggregate_commute_times
from methods._fosr import apply_fosr
from torch_geometric.datasets import ZINC, QM9
from torch_geometric.utils import to_networkx
from methods import _fosr
import numpy as np
import networkx as nx
from math import inf
from numba import jit, int64


def process_dataset(dataset_name):
    if dataset_name.lower() == 'zinc':
        dataset = ZINC(root='/tmp/ZINC', subset=False)  # Using the full dataset
    elif dataset_name.lower() == 'qm9':
        dataset = QM9(root='/tmp/QM9')
    else:
        raise ValueError("Invalid dataset name. Choose 'ZINC' or 'QM9'.")
    
    total_original_edges = 0
    total_fosr_edges = 0
    total_original_commute_time = 0
    total_fosr_commute_time = 0
    total_graphs = len(dataset)
    
    for i, data in enumerate(dataset):
        # Convert to NetworkX graph
        G_original = to_networkx(data, to_undirected=True)
        
        # Get edge index from the graph
        edge_index = np.array(list(G_original.edges())).T
        
        # Calculate max new edges (10% of original edges)
        max_new_edges = int(0.1 * G_original.number_of_edges())
        
        # Apply FOSR to get new edge index
        new_edge_index, _ = apply_fosr(edge_index, max_new_edges)
        
        # Create a new graph with the rewired edges
        G_fosr = nx.Graph()
        G_fosr.add_nodes_from(range(data.num_nodes))
        new_edges = list(zip(new_edge_index[0], new_edge_index[1]))
        G_fosr.add_edges_from(new_edges)
        
        # Calculate commute times
        original_commute_time = aggregate_commute_times(G_original)
        fosr_commute_time = aggregate_commute_times(G_fosr)
        
        total_original_edges += G_original.number_of_edges()
        total_fosr_edges += G_fosr.number_of_edges()
        total_original_commute_time += original_commute_time
        total_fosr_commute_time += fosr_commute_time
        
        if (i + 1) % 10000 == 0:
            print(f"Processed {i+1}/{total_graphs} graphs...")
    
    return total_original_edges, total_fosr_edges, total_original_commute_time, total_fosr_commute_time, total_graphs

# Function to print summary statistics
def print_summary_to_file(file, original_edges, fosr_edges, original_commute_time, fosr_commute_time, total_graphs, dataset_name):
    summary = (
        f"\nSummary for {dataset_name} dataset:\n"
        f"Total graphs processed: {total_graphs}\n"
        f"Number of original edges: {original_edges}\n"
        f"Number of edges in FOSR dataset: {fosr_edges}\n"
        f"Difference in edges: {fosr_edges - original_edges}\n"
        f"Percentage of edges added: {((fosr_edges - original_edges) / original_edges) * 100:.2f}%\n"
        f"Average original commute time: {original_commute_time / total_graphs:.4f}\n"
        f"Average FOSR commute time: {fosr_commute_time / total_graphs:.4f}\n"
        f"Percentage change in commute time: {((fosr_commute_time - original_commute_time) / original_commute_time) * 100:.2f}%\n"
    )

    # Print to console
    print(summary)

    # Write the summary to the file
    file.write(summary)

# Create the results folder if it doesn't exist
os.makedirs("results", exist_ok=True)

# Open the results file in write mode
with open("results/exp1_results.txt", "w") as f:

    # Process ZINC dataset
    print("Processing ZINC dataset...")
    zinc_original, zinc_fosr, zinc_original_ct, zinc_fosr_ct, zinc_graphs = process_dataset('ZINC')

    # Process QM9 dataset
    print("\nProcessing QM9 dataset...")
    qm9_original, qm9_fosr, qm9_original_ct, qm9_fosr_ct, qm9_graphs = process_dataset('QM9')

    # Print summaries to console and file
    print_summary_to_file(f, zinc_original, zinc_fosr, zinc_original_ct, zinc_fosr_ct, zinc_graphs, "ZINC")
    print_summary_to_file(f, qm9_original, qm9_fosr, qm9_original_ct, qm9_fosr_ct, qm9_graphs, "QM9")