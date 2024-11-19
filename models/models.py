import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GINConv, global_add_pool, BatchNorm, global_mean_pool


class GINModel(torch.nn.Module):
    def __init__(self, num_features, num_classes=1, hidden_dim=64, depth=3):
        super(GINModel, self).__init__()

        # Define GIN layers
        self.convs = torch.nn.ModuleList()
        self.convs.append(GINConv(
            torch.nn.Sequential(
                torch.nn.Linear(num_features, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU()
            )
        ))

        # Additional GIN layers
        for _ in range(depth - 1):
            self.convs.append(GINConv(
                torch.nn.Sequential(
                    torch.nn.Linear(hidden_dim, hidden_dim),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_dim, hidden_dim),
                    torch.nn.ReLU()
                )
            ))

        # Batch normalization layers
        self.batch_norms = torch.nn.ModuleList([BatchNorm(hidden_dim) for _ in range(depth)])

        # Final regression layer
        self.final_lin = torch.nn.Linear(hidden_dim, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = conv(x, edge_index)
            x = batch_norm(x)
            x = F.relu(x)
        x = global_add_pool(x, data.batch)  # Pool to get a graph-level representation
        x = self.final_lin(x)  # Final regression output
        return x


class GATModel(nn.Module):
    def __init__(self, num_features, num_classes=1, hidden_dim=64, depth=3):
        super(GATModel, self).__init__()
        self.layers = nn.ModuleList()

        # First GAT layer
        self.layers.append(GATConv(num_features, hidden_dim, heads=4, concat=True))

        # Hidden GAT layers based on depth
        for _ in range(1, depth - 1):
            self.layers.append(GATConv(hidden_dim * 4, hidden_dim, heads=4, concat=True))

        # Final GAT layer
        self.layers.append(GATConv(hidden_dim * 4, hidden_dim, heads=4, concat=True))

        # Fully-connected layer for output
        self.fc = nn.Linear(hidden_dim * 4, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Apply each GAT layer
        for conv in self.layers:
            x = conv(x, edge_index)

        # Global pooling over nodes in each graph in the batch
        x = global_add_pool(x, batch)

        # Final classification layer
        x = self.fc(x)

        return x
    

class GAT_zinc(torch.nn.Module):
    def __init__(self, num_features, num_classes=1, hidden_dim=64, depth=3, heads=8):
        super(GAT_zinc, self).__init__()
        
        # Initialize GAT layers
        self.convs = torch.nn.ModuleList()
        # First GAT layer
        self.convs.append(GATConv(num_features, hidden_dim, heads=heads))
        # Hidden GAT layers
        for _ in range(depth - 1):
            self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads))
        
        # Linear layer for the final output
        self.lin = torch.nn.Linear(hidden_dim * heads, num_classes)  # Predicting a scalar value (regression)

    def forward(self, batch):
        # Extract components from the batch object
        x, edge_index, batch_vector = batch.x, batch.edge_index, batch.batch

        # Pass through GAT layers
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.elu(x)  # Use ELU activation after each GAT layer
            x = F.dropout(x, p=0.5, training=self.training)  # Dropout for regularization
        
        # Global mean pooling to aggregate node features into graph-level features
        x = global_mean_pool(x, batch_vector)
        
        # Final linear layer for regression
        x = self.lin(x)
        return x
