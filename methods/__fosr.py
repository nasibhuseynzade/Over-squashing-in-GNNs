def apply_fosr_to_qm9(dataset, num_iterations=100):
    all_new_edge_indices = []
    all_new_edge_types = []

    for data in tqdm(dataset, desc="Applying FOSR to molecules"):
        # Convert to undirected graph
        edge_index = to_undirected(data.edge_index).numpy()
        
        # Initialize edge_type and node features x
        edge_type = np.zeros(edge_index.shape[1], dtype=np.int64)
        n = np.max(edge_index) + 1
        x = 2 * np.random.random(n) - 1
        
        # Perform FOSR rewire
        new_edge_index, new_edge_type, _ = edge_rewire(
            edge_index,
            x=x,
            edge_type=edge_type,
            num_iterations=num_iterations,
            initial_power_iters=5
        )

        # Append results
        all_new_edge_indices.append(torch.tensor(new_edge_index, dtype=torch.long))
        all_new_edge_types.append(torch.tensor(new_edge_type, dtype=torch.long))

    return all_new_edge_indices, all_new_edge_types