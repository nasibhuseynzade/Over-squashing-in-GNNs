import torch
import torch.nn.functional as F
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, GINConv, global_mean_pool
import matplotlib.pyplot as plt
from torch_geometric.utils import add_self_loops

# Load QM9 dataset
dataset = QM9(root='data/QM9')
dataset = dataset.shuffle()
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define Models
class BasicGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(BasicGNN, self).__init__()
        self.linear1 = torch.nn.Linear(in_channels, hidden_channels)
        self.linear2 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.out_layer = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        x = self.message_passing(x, edge_index)
        x = F.relu(self.linear1(x))

        x = self.message_passing(x, edge_index)
        x = F.relu(self.linear2(x))

        x = global_mean_pool(x, batch)
        return self.out_layer(x)

    def message_passing(self, x, edge_index):
        row, col = edge_index
        aggr_out = torch.zeros_like(x)
        aggr_out.index_add_(0, row, x[col])
        out = x + aggr_out
        return out

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=4, concat=True)
        self.conv2 = GATConv(4 * hidden_channels, hidden_channels, heads=1, concat=False)
        self.lin = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        x = F.elu(self.conv1(x, edge_index))
        x = F.elu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.lin(x)

class GIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GIN, self).__init__()
        nn = torch.nn.Sequential(torch.nn.Linear(in_channels, hidden_channels), torch.nn.ReLU(), torch.nn.Linear(hidden_channels, hidden_channels))
        self.conv1 = GINConv(nn)
        self.conv2 = GINConv(nn)
        self.lin = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.lin(x)

# R2 Score Function
def r2_score(y_true, y_pred):
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot

# Training and Evaluation Functions with R2 Score
def train(model, loader, optimizer):
    model.train()
    total_r2 = 0
    for data in loader:
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = F.mse_loss(out, data.y)
        loss.backward()
        optimizer.step()

        # Calculate R2 score for each batch and accumulate
        total_r2 += r2_score(data.y, out) * data.num_graphs
    return total_r2 / len(loader.dataset)

def test(model, loader):
    model.eval()
    total_r2 = 0
    with torch.no_grad():
        for data in loader:
            out = model(data.x, data.edge_index, data.batch)
            total_r2 += r2_score(data.y, out) * data.num_graphs
    return total_r2 / len(loader.dataset)

# Main Experiment Loop
def run_experiment():
    epochs = 100
    hidden_channels = 64
    out_channels = 1  # For regression (modify if classification)

    models = {
        'BasicGNN': BasicGNN(dataset.num_node_features, hidden_channels, out_channels),
        'GAT': GAT(dataset.num_node_features, hidden_channels, out_channels),
        'GIN': GIN(dataset.num_node_features, hidden_channels, out_channels)
    }
    
    histories = {name: {'train_r2': [], 'val_r2': []} for name in models.keys()}

    for name, model in models.items():
        print(f'\nTraining {name} model...')
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        for epoch in range(epochs):
            train_r2 = train(model, train_loader, optimizer)
            val_r2 = test(model, val_loader)
            histories[name]['train_r2'].append(train_r2)
            histories[name]['val_r2'].append(val_r2)
            
            if (epoch + 1) % 10 == 0:
                print(f'{name} Epoch {epoch + 1}, Train R2: {train_r2:.4f}, Val R2: {val_r2:.4f}')

    return histories

# Plotting R2 Results
histories = run_experiment()

plt.figure(figsize=(12, 6))
for model_name, history in histories.items():
    plt.plot(history['train_r2'], label=f'{model_name} Train R2')
    plt.plot(history['val_r2'], linestyle='--', label=f'{model_name} Val R2')
plt.xlabel('Epoch')
plt.ylabel('R² Score')
plt.legend()
plt.title('Training and Validation R² Score Across Models')
plt.show()
