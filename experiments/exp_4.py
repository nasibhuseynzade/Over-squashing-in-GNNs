import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_mean_pool
from torch_geometric.utils import to_networkx
from sklearn.metrics import accuracy_score

class RGINConv(torch.nn.Module):
    def __init__(self, in_features, out_features, num_relations):
        super(RGINConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_relations = num_relations
        self.self_loop_conv = torch.nn.Linear(in_features, out_features)
        convs = []
        for i in range(self.num_relations):
            convs.append(GINConv(nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features),
                nn.ReLU(),
                nn.Linear(out_features, out_features)
            )))
        self.convs = ModuleList(convs)

    def forward(self, x, edge_index, edge_type):
        x_new = self.self_loop_conv(x)
        for i, conv in enumerate(self.convs):
            mask = (edge_type == i)
            if mask.any():
                # Correctly index edge_index using the mask
                rel_edge_index = edge_index[:, mask]
                x_new = x_new + conv(x, rel_edge_index)
        return x_new

class RGINQM9(nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_relations, num_tasks):
        super(RGINQM9, self).__init__()
        self.conv1 = RGINConv(num_node_features, hidden_channels, num_relations)
        self.conv2 = RGINConv(hidden_channels, hidden_channels, num_relations)
        self.conv3 = RGINConv(hidden_channels, hidden_channels, num_relations)
        self.lin = nn.Linear(hidden_channels, num_tasks)

    def forward(self, data):
        x, edge_index, edge_type, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # Convert edge_type to long tensor and get the first column if it's 2D
        if edge_type.dim() > 1:
            edge_type = edge_type[:, 0]
        edge_type = edge_type.long()

        x = F.relu(self.conv1(x, edge_index, edge_type))
        x = F.relu(self.conv2(x, edge_index, edge_type))
        x = F.relu(self.conv3(x, edge_index, edge_type))

        x = global_mean_pool(x, batch)
        x = self.lin(x)

        return x
        
# Load QM9 dataset
dataset = QM9(root='path/to/qm9')
train_loader = DataLoader(dataset[:int(len(dataset)*0.8)], batch_size=32, shuffle=True)
test_loader = DataLoader(dataset[int(len(dataset)*0.8):], batch_size=32, shuffle=False)

def train_and_evaluate(model, train_loader, test_loader, num_epochs=100, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()  # Use MSE loss for regression tasks in QM9

    for epoch in range(num_epochs):
        model.train()
        for data in train_loader:
            optimizer.zero_grad()
            out = model(data)
            loss = loss_fn(out, data.y)  # Assuming `data.y` contains the target
            loss.backward()
            optimizer.step()

    # Evaluation
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for data in test_loader:
            out = model(data)
            predictions.append(out)
            actuals.append(data.y)

    # Compute accuracy (or any other metric)
    predictions = torch.cat(predictions).numpy()
    actuals = torch.cat(actuals).numpy()
    accuracy = accuracy_score(np.argmax(actuals, axis=1), np.argmax(predictions, axis=1))
    return accuracy

num_iterations_list = [int(0.005 * dataset.num_nodes),  # 0.5% of nodes
                       int(0.01 * dataset.num_nodes),   # 1% of nodes
                       int(0.02 * dataset.num_nodes),   # 2% of nodes
                       int(0.05 * dataset.num_nodes),   # 5% of nodes
                       int(0.1 * dataset.num_nodes)]     # 10% of nodes

accuracies = []

for num_iterations in num_iterations_list:
    model = RGINQM9(num_node_features=dataset.num_node_features,
                    hidden_channels=64,
                    num_relations=dataset.num_edge_features,
                    num_tasks=dataset.num_classes)

    accuracy = train_and_evaluate(model, train_loader, test_loader, num_epochs=100, lr=0.001)
    accuracies.append(accuracy)

plt.figure(figsize=(10, 6))
plt.plot(num_iterations_list, accuracies, marker='o')
plt.xscale('log')
plt.xticks(num_iterations_list, [f"{n} ({p:.1f}%)" for n, p in zip(num_iterations_list, [0.5, 1, 2, 5, 10])])
plt.xlabel('Number of Iterations (as percentage of nodes)')
plt.ylabel('Accuracy')
plt.title('Effect of Number of Iterations on Model Accuracy')
plt.grid()
plt.show()
