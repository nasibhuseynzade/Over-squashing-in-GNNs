from torch_geometric.datasets import ZINC

# Load ZINC dataset
dataset = ZINC(root='/Users/nasibhuseynzade/Downloads/ZINC_dataset', subset=True)  # Use subset for smaller size
print(f"Dataset size: {len(dataset)}")
print(dataset[0])  # Print the first graph
