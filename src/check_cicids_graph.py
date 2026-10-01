import numpy as np
from pathlib import Path

root = Path(__file__).resolve().parent.parent

path = root / "data" / "graphs_cicids" / "train_graph_00000.npz"

graph = np.load(path)

print("Keys:", graph.files)

print("Nodes:", graph["num_nodes"])
print("Edge index:", graph["edge_index"].shape)
print("Edge features:", graph["edge_attr"].shape)
print("Edge labels:", graph["edge_label"].shape)

print("\nNaN:", np.isnan(graph["edge_attr"]).sum())
print("Inf:", np.isinf(graph["edge_attr"]).sum())

print("\nBenign edges:", np.sum(graph["edge_label"] == 0))
print("Malicious edges:", np.sum(graph["edge_label"] == 1))

print("\nFirst 5 edges:")
print(graph["edge_index"][:, :5])

print("\nFirst edge feature shape:")
print(graph["edge_attr"][0].shape)