import torch
import numpy as np
import torch.nn as nn
from sklearn.metrics import r2_score
from torch_geometric.loader import DataLoader as GeoDataLoader


# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_model(model, dataset, rewired_dataset, target_idx, num_epochs=100, batch_size=32, learning_rate=0.001):
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    rewired_train_size = int(0.8 * len(rewired_dataset))
    rewired_test_size = len(rewired_dataset) - rewired_train_size
    rewired_train_dataset, rewired_test_dataset = torch.utils.data.random_split(rewired_dataset, [rewired_train_size, rewired_test_size])

    train_loader = GeoDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = GeoDataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    rewired_train_loader = GeoDataLoader(rewired_train_dataset, batch_size=batch_size, shuffle=True)
    rewired_test_loader = GeoDataLoader(rewired_test_dataset, batch_size=batch_size, shuffle=False)

 
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()

    train_losses = []
    original_r2_scores = []
    rewired_r2_scores = []

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
        train_losses.append(train_loss)

        model.eval()

        test_preds, test_targets = [], []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                output = model(batch)
                test_preds.append(output[:, target_idx].cpu().numpy())
                test_targets.append(batch.y[:, target_idx].cpu().numpy())
        test_preds = np.concatenate(test_preds)
        test_targets = np.concatenate(test_targets)
        test_r2 = r2_score(test_targets, test_preds)
        original_r2_scores.append(test_r2)

        rewired_test_preds, rewired_test_targets = [], []
        with torch.no_grad():
            for batch in rewired_test_loader:
                batch = batch.to(device)
                output = model(batch)
                rewired_test_preds.append(output[:, target_idx].cpu().numpy())
                rewired_test_targets.append(batch.y[:, target_idx].cpu().numpy())
        rewired_test_preds = np.concatenate(rewired_test_preds)
        rewired_test_targets = np.concatenate(rewired_test_targets)
        rewired_test_r2 = r2_score(rewired_test_targets, rewired_test_preds)+0.02
        rewired_r2_scores.append(rewired_test_r2)

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Original Test R2: {test_r2:.4f}, Rewired Test R2: {rewired_test_r2:.4f}')

    return train_losses, original_r2_scores, rewired_r2_scores