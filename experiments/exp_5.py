import torch
import torch.nn as nn
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.data import DataLoader
from torch_geometric.datasets import QM9
import numpy as np
from sklearn.metrics import r2_score

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# GAT Model
class GATModel(nn.Module):
    def __init__(self, num_features, num_classes):
        super(GATModel, self).__init__()
        
        self.conv1 = GATConv(num_features, 64, heads=4, concat=True)
        self.conv2 = GATConv(64 * 4, 64, heads=4, concat=True)
        self.conv3 = GATConv(64 * 4, 64, heads=4, concat=True)
        self.fc = nn.Linear(64 * 4, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        x = self.conv3(x, edge_index)
        
        x = global_add_pool(x, batch)
        x = self.fc(x)
        
        return x

# GIN Model
class GINModel(nn.Module):
    def __init__(self, num_features, num_classes):
        super(GINModel, self).__init__()
        
        self.conv1 = GINConv(nn.Sequential(
            nn.Linear(num_features, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        ))
        self.conv2 = GINConv(nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        ))
        self.conv3 = GINConv(nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        ))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        x = self.conv3(x, edge_index)
        
        x = global_add_pool(x, batch)
        x = self.fc(x)
        
        return x

# Train and evaluate the GAT model on QM9 dataset
def train_model(dataset, target_idx, num_epochs=100, batch_size=32, learning_rate=0.001):
    # Split the dataset into train and test sets
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model
    model = GATModel(num_features=dataset.num_features, num_classes=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # Train the model
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output[:, target_idx], batch.y[:, target_idx])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_loss = total_loss / len(train_loader)

        # Evaluate on the test set
        model.eval()
        test_preds = []
        test_targets = []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                output = model(batch)
                test_preds.append(output[:, target_idx].cpu().numpy())
                test_targets.append(batch.y[:, target_idx].cpu().numpy())
        test_preds = np.concatenate(test_preds)
        test_targets = np.concatenate(test_targets)
        test_r2 = r2_score(test_targets, test_preds)

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test R2: {test_r2:.4f}')

    return test_r2

# Load the QM9 dataset
dataset = QM9(root='qm9_data')

# Train the GAT model and get the R2 score
r2_score = train_model(dataset, target_idx=0)
print(f'Final R2 score: {r2_score:.4f}')

# Train the GIN model and get the R2 score
r2_score = train_model(dataset, target_idx=0)
print(f'Final R2 score: {r2_score:.4f}')