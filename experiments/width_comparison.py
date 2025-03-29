import os, sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CURRENT_DIR))

import torch
import matplotlib.pyplot as plt
import pickle
import numpy as np

from models_and_training.training import _train_model_new
from models_and_training.models import GATModel, GINModel, GCNModel

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

try:
    with open('/home/huseynzade/SRP/datasets/qm9_dataset.pkl', 'rb') as f:
        dataset = pickle.load(f)
except FileNotFoundError:
    with open('/Users/nasibhuseynzade/Downloads/qm9_dataset.pkl','rb') as f:
        dataset = pickle.load(f)

plt.figure(figsize=(10, 6))

hidden_dims = [64, 128, 256]  # Different hidden dimensions
num_epochs = 100
num_runs = 5  # Number of runs per hidden dimension

# Dictionary to store results
results = {}

for hidden_dim in hidden_dims:
    print(f"\nTraining model for Hidden Dimension = {hidden_dim}\n" + "="*80)
    
    # Store results for this hidden dimension
    gin_r2_all = np.zeros((num_runs, num_epochs))
    gat_r2_all = np.zeros((num_runs, num_epochs))
    gcn_r2_all = np.zeros((num_runs, num_epochs))

    for run in range(num_runs):
        print(f"\nRun {run+1}/{num_runs}: Training model with hidden dimension = {hidden_dim}\n" + "-"*80)

        model1 = GINModel(num_features=dataset.num_features, num_classes=1, hidden_dim=hidden_dim).to(device)
        model2 = GATModel(num_features=dataset.num_features, num_classes=1, hidden_dim=hidden_dim).to(device)
        model3 = GCNModel(num_features=dataset.num_features, num_classes=1, hidden_dim=hidden_dim).to(device)

        # Train the model
        r2_gin = _train_model_new(model1, dataset, batch_size=64, learning_rate=0.001, target_idx=0, num_epochs=num_epochs)
        r2_gat = _train_model_new(model2, dataset, batch_size=64, learning_rate=0.001, target_idx=0, num_epochs=num_epochs)
        r2_gcn = _train_model_new(model3, dataset, batch_size=64, learning_rate=0.001, target_idx=0, num_epochs=num_epochs)

        # Store results
        gin_r2_all[run, :] = r2_gin
        gat_r2_all[run, :] = r2_gat
        gcn_r2_all[run, :] = r2_gcn

        # Print individual run results
        print(f"Run {run+1} - Final GIN R²: {r2_gin[-1]:.4f}, Final GAT R²: {r2_gat[-1]:.4f}, Final GCN R²: {r2_gcn[-1]:.4f}")

    # Compute mean and standard deviation
    gin_r2_mean = np.mean(gin_r2_all, axis=0)
    gin_r2_std = np.std(gin_r2_all, axis=0)
    gat_r2_mean = np.mean(gat_r2_all, axis=0)
    gat_r2_std = np.std(gat_r2_all, axis=0)
    gcn_r2_mean = np.mean(gcn_r2_all, axis=0)
    gcn_r2_std = np.std(gcn_r2_all, axis=0)

    # Store results in dictionary
    results[hidden_dim] = {
        "gin_r2_mean": gin_r2_mean[-1],
        "gin_r2_std": gin_r2_std[-1],
        "gat_r2_mean": gat_r2_mean[-1],
        "gat_r2_std": gat_r2_std[-1],
        "gcn_r2_mean": gcn_r2_mean[-1],
        "gcn_r2_std": gcn_r2_std[-1]
    }

# Print final summary results
print("\nFinal Summary Results After 5 Runs for Each Hidden Dimension:")
for hidden_dim in hidden_dims:
    print(f"\nHidden Dimension = {hidden_dim}")
    print(f"Final GIN R² Mean: {results[hidden_dim]['gin_r2_mean']:.4f}, Std Dev: {results[hidden_dim]['gin_r2_std']:.4f}")
    print(f"Final GAT R² Mean: {results[hidden_dim]['gat_r2_mean']:.4f}, Std Dev: {results[hidden_dim]['gat_r2_std']:.4f}")
    print(f"Final GCN R² Mean: {results[hidden_dim]['gcn_r2_mean']:.4f}, Std Dev: {results[hidden_dim]['gcn_r2_std']:.4f}")
