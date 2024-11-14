# optimizing hyperparameter of fosr method
import os, sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CURRENT_DIR))

import torch
from torch_geometric.datasets import QM9
import numpy as np
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import torch.nn as nn
import optuna
import matplotlib.pyplot as plt
import seaborn as sns

from preprocessing.fosr import edge_rewire
from models.models import GINModel

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Train and evaluate the GIN model on QM9 dataset
def train_gin_model(dataset, target_idx, num_epochs=100, batch_size=32, learning_rate=0.001):
    # Split the dataset into train and test sets
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model
    model = GINModel(num_features=dataset.num_features, num_classes=1).to(device)
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

# Define the objective function for Optuna
def objective(trial):
    # Suggest a value for `num_iterations` in the range [1, 100]
    num_iterations = trial.suggest_int("num_iterations", 1, 100)
    dataset = QM9(root='qm9_data')
    # Create a copy of the original dataset
    fosr_dataset = dataset.copy()

    # Apply edge re-wiring with the selected number of iterations
    for i in range(len(dataset)):
        edge_index, edge_type, _ = edge_rewire(dataset[i].edge_index.numpy(), num_iterations=num_iterations)
        fosr_dataset[i].edge_index = torch.tensor(edge_index)
        fosr_dataset[i].edge_type = torch.tensor(edge_type)

    # Train the model on the rewired dataset and get the test R2 score
    test_r2 = train_gin_model(fosr_dataset, target_idx=0, num_epochs=5, batch_size=32, learning_rate=0.001)

    return test_r2

# Set up and run the Optuna study
study = optuna.create_study(direction="maximize")  # Maximize the R2 score
study.optimize(objective, n_trials=50)  # Run 50 trials (can be adjusted)

# Print the best value for `num_iterations` and the corresponding R2 score
print("Best num_iterations:", study.best_trial.params["num_iterations"])
print("Best Test R2:", study.best_value)


# After the study is complete, get a dataframe of the trial results
results_df = study.trials_dataframe(attrs=('params', 'value'))

# Rename the columns for easier access
results_df.rename(columns={'params_num_iterations': 'num_iterations', 'value': 'r2_score'}, inplace=True)
# Plot the results
plt.figure(figsize=(10, 6))
sns.scatterplot(data=results_df, x='num_iterations', y='r2_score', s=60, color="b", alpha=0.7)
plt.title("R² Score vs. num_iterations")
plt.xlabel("num_iterations")
plt.ylabel("R² Score")
plt.xlim(1, 100)
plt.grid(True)


results_folder='/Users/nasibhuseynzade/Desktop/SRP_code/results'
# Save the plot to the results folder before showing it
plot_path = os.path.join(results_folder, 'r2_score_optimization.png')
plt.savefig(plot_path, format='png')

# Show the plot in the console (optional)
plt.show()

# Specify the path for the results text file
results_text_path = os.path.join(results_folder, 'exp7_results.txt')

# Create the text file and write data
with open(results_text_path, 'w') as file:
    file.write("Experiment 7 Results\n")
    file.write(results_df.to_string(index=False))  # Save DataFrame content as text Explanation