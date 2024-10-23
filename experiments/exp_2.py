import torch
import torch.nn.functional as F
from torch_geometric.datasets import QM9
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv, GATConv, GINConv, global_add_pool
import matplotlib.pyplot as plt

# Load QM9 dataset
dataset = QM9(root='./data/QM9')
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Define GNN (GCN) Model
class GNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_add_pool(x, data.batch)  # Pooling layer
        x = self.fc(x)
        return x

# Define GAT Model
class GAT(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GAT, self).__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=8, concat=True)
        self.conv2 = GATConv(hidden_dim * 8, hidden_dim, heads=1, concat=True)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_add_pool(x, data.batch)
        x = self.fc(x)
        return x

# Define GIN Model
class GIN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GIN, self).__init__()
        self.conv1 = GINConv(torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim)
        ))
        self.conv2 = GINConv(torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim)
        ))
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_add_pool(x, data.batch)
        x = self.fc(x)
        return x

# Training Function
def train(model, loader, optimizer, criterion, epochs=100):
    model.train()
    all_losses = []
    for epoch in range(epochs):
        total_loss = 0
        for data in loader:
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        all_losses.append(avg_loss)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}')
    return all_losses

# Main experiment function
def run_experiment():
    input_dim = dataset.num_features
    hidden_dim = 64
    output_dim = 1  # Assuming a single target property for simplicity
    
    # Initialize models
    gnn_model = GNN(input_dim, hidden_dim, output_dim)
    gat_model = GAT(input_dim, hidden_dim, output_dim)
    gin_model = GIN(input_dim, hidden_dim, output_dim)

    # Optimizers and Loss
    optimizer_gnn = torch.optim.Adam(gnn_model.parameters(), lr=0.001)
    optimizer_gat = torch.optim.Adam(gat_model.parameters(), lr=0.001)
    optimizer_gin = torch.optim.Adam(gin_model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()  # Use MSE for regression tasks, or CrossEntropy for classification
    
    # Train each model
    print("Training GNN Model...")
    gnn_losses = train(gnn_model, loader, optimizer_gnn, criterion)
    
    print("Training GAT Model...")
    gat_losses = train(gat_model, loader, optimizer_gat, criterion)
    
    print("Training GIN Model...")
    gin_losses = train(gin_model, loader, optimizer_gin, criterion)

    # Plot Loss Graph
    plt.figure(figsize=(10, 6))
    plt.plot(gnn_losses, label='GNN Loss')
    plt.plot(gat_losses, label='GAT Loss')
    plt.plot(gin_losses, label='GIN Loss')
    plt.title('Loss Curves for GNN, GAT, and GIN Models')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# Run the experiment
run_experiment()
