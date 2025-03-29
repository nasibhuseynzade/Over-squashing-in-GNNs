import torch
import numpy as np
from sklearn.metrics import r2_score
from torch_geometric.loader import DataLoader as GeoDataLoader


# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_model(
    model, 
    original_dataset, 
    fosr_dataset, 
    sdrf_dataset, 
    target_idx, 
    num_epochs=100, 
    batch_size=64, 
    learning_rate=0.0005
): #for comparing two datasets
    def prepare_loaders(dataset, batch_size):
        """Utility to split dataset and create data loaders."""
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
        train_loader = GeoDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = GeoDataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, test_loader
    
    # Prepare loaders for the datasets
    original_train_loader, original_test_loader = prepare_loaders(original_dataset, batch_size)
    _, fosr_test_loader = prepare_loaders(fosr_dataset, batch_size)
    _, sdrf_test_loader = prepare_loaders(sdrf_dataset, batch_size)

    # Optimizer and loss criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()

    original_r2_scores = []
    fosr_r2_scores = []
    sdrf_r2_scores = []

    for epoch in range(num_epochs):
        model.train()

        # Training on the original dataset
        for batch in original_train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output[:, target_idx], batch.y[:, target_idx])
            loss.backward()
            optimizer.step()
        model.eval()

        # Evaluate on Original Dataset
        test_preds, test_targets = [], []
        with torch.no_grad():
            for batch in original_test_loader:
                batch = batch.to(device)
                output = model(batch)
                test_preds.append(output[:, target_idx].cpu().numpy())
                test_targets.append(batch.y[:, target_idx].cpu().numpy())
        test_preds = np.concatenate(test_preds)
        test_targets = np.concatenate(test_targets)
        original_r2 = r2_score(test_targets, test_preds)
        original_r2_scores.append(original_r2)

        # Evaluate on FOSR Dataset
        fosr_preds, fosr_targets = [], []
        with torch.no_grad():
            for batch in fosr_test_loader:
                batch = batch.to(device)
                output = model(batch)
                fosr_preds.append(output[:, target_idx].cpu().numpy())
                fosr_targets.append(batch.y[:, target_idx].cpu().numpy())
        fosr_preds = np.concatenate(fosr_preds)
        fosr_targets = np.concatenate(fosr_targets)
        fosr_r2 = r2_score(fosr_targets, fosr_preds)
        fosr_r2_scores.append(fosr_r2)

        # Evaluate on SDRF Dataset
        sdrf_preds, sdrf_targets = [], []
        with torch.no_grad():
            for batch in sdrf_test_loader:
                batch = batch.to(device)
                output = model(batch)
                sdrf_preds.append(output[:, target_idx].cpu().numpy())
                sdrf_targets.append(batch.y[:, target_idx].cpu().numpy())
        sdrf_preds = np.concatenate(sdrf_preds)
        sdrf_targets = np.concatenate(sdrf_targets)
        sdrf_r2 = r2_score(sdrf_targets, sdrf_preds)
        sdrf_r2_scores.append(sdrf_r2)

        print(f"Epoch {epoch+1}/{num_epochs}"
              f"Original Test R2: {original_r2:.4f}, FOSR Test R2: {fosr_r2:.4f}, SDRF Test R2: {sdrf_r2:.4f}")

    return original_r2_scores, fosr_r2_scores, sdrf_r2_scores


def _train_model_new(model, dataset, target_idx, num_epochs=100, batch_size=32, learning_rate=0.001): # for comparing only two datasets
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = GeoDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = GeoDataLoader(test_dataset, batch_size=batch_size, shuffle=False)
 
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()

    train_losses = []
    original_r2_scores = []

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

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Original Test R2: {test_r2:.4f}')

    return original_r2_scores

def train_test_model(model, dataset, rewired_dataset, num_epochs=4, batch_size=32, learning_rate=0.0005):  #for zinc dataset

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    rewired_train_size = int(0.8 * len(rewired_dataset))
    rewired_test_size = len(rewired_dataset) - rewired_train_size
    _, rewired_test_dataset = torch.utils.data.random_split(rewired_dataset, [rewired_train_size, rewired_test_size])

    train_loader = GeoDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = GeoDataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    rewired_test_loader = GeoDataLoader(rewired_test_dataset, batch_size=batch_size, shuffle=False)



    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    original_r2_scores = []
    rewired_r2_scores = []

    for epoch in tqdm(range(num_epochs)):
 
        model.train()
        total_loss = 0

        for batch in train_loader:
            batch = batch.to(device)
            batch.x = batch.x.float()
            batch.y = batch.y.float().view(-1, 1)
            optimizer.zero_grad()
            out = model(batch)  
            
            loss = F.mse_loss(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs

        train_loss = total_loss / len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        test_preds, test_targets = [], []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                batch.x = batch.x.float()
                batch.y = batch.y.float().view(-1, 1)
            
                out = model(batch)
                test_targets.append(batch.y.cpu().numpy())
                test_preds.append(out.cpu().numpy())
    
        test_targets = np.concatenate(test_targets)
        test_preds = np.concatenate(test_preds)
        test_r2 = r2_score(test_targets, test_preds)
        if test_r2 < -0.1:
            test_r2 = -0.1
        original_r2_scores.append(test_r2)

        rewired_test_preds, rewired_test_targets = [], []
        with torch.no_grad():
            for batch in rewired_test_loader:
                batch = batch.to(device)
                batch.x = batch.x.float()
                batch.y = batch.y.float().view(-1, 1)
                out = model(batch)
                rewired_test_preds.append(batch.y.cpu().numpy())
                rewired_test_targets.append(out.cpu().numpy())

        rewired_test_preds = np.concatenate(rewired_test_preds)
        rewired_test_targets = np.concatenate(rewired_test_targets)
        rewired_test_r2 = r2_score(rewired_test_targets, rewired_test_preds)
        if rewired_test_r2 < -0.1:
            rewired_test_r2 = -0.1
        rewired_r2_scores.append(rewired_test_r2)

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Original Test R2: {test_r2:.4f}, Rewired Test R2: {rewired_test_r2:.4f}')

    
    return train_losses, original_r2_scores, rewired_r2_scores

