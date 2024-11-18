def commute_time_rewiring(
    data,
    loops=10,
    remove_edges=True,
    removal_bound=None,
    is_undirected=False
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
        
    current_commute_time = commute_time(G)

    for _ in range(loops):
        can_add = False
        best_improvement = 0
        best_edge = None
        
        # Try adding edges between all possible node pairs
        for x in range(N):
            for y in range(N):
                if x != y and not G.has_edge(x, y):
                    # Temporarily add edge
                    G.add_edge(x, y)
                    if is_undirected:
                        A[x, y] = A[y, x] = 1
                    else:
                        A[x, y] = 1
                        
                    # Calculate new commute time
                    new_commute_time = commute_time(G)
                    improvement = current_commute_time - new_commute_time
                    
                    # Remove temporary edge
                    G.remove_edge(x, y)
                    if is_undirected:
                        A[x, y] = A[y, x] = 0
                    else:
                        A[x, y] = 0
                    
                    # Update best improvement
                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_edge = (x, y)
                        can_add = True
        
        # Add the best edge if found
        if can_add:
            x, y = best_edge
            G.add_edge(x, y)
            if is_undirected:
                A[x, y] = A[y, x] = 1
            else:
                A[x, y] = 1
            edge_type = np.append(edge_type, 1)
            if is_undirected:
                edge_type = np.append(edge_type, 1)
            current_commute_time = commute_time(G)
            
        # Optional edge removal based on commute time improvement
        if remove_edges:
            best_removal_improvement = 0
            edge_to_remove = None
            
            # Try removing each existing edge
            for edge in G.edges():
                x, y = edge
                G.remove_edge(x, y)
                new_commute_time = commute_time(G)
                improvement = current_commute_time - new_commute_time
                
                # Restore edge
                G.add_edge(x, y)
                
                if improvement > best_removal_improvement:
                    best_removal_improvement = improvement
                    edge_to_remove = (x, y)
            
            # Remove edge if it improves commute time
            if edge_to_remove and (removal_bound is None or best_removal_improvement > removal_bound):
                x, y = edge_to_remove
                G.remove_edge(x, y)
                if is_undirected:
                    A[x, y] = A[y, x] = 0
                else:
                    A[x, y] = 0
                current_commute_time = commute_time(G)
        
        # Break if no more improvements can be made
        if not can_add and (not remove_edges or best_removal_improvement == 0):
            break
            
    return from_networkx(G).edge_index, torch.tensor(edge_type)