import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path

root = Path(__file__).resolve().parent.parent

graph = np.load(
    root / "data" / "graphs_cicids" / "train_graph_00000.npz"
)

edge_index = graph["edge_index"]
edge_label = graph["edge_label"]

G = nx.DiGraph()

num_edges = 100

for i in range(num_edges):
    src = int(edge_index[0, i])
    dst = int(edge_index[1, i])
    G.add_edge(src, dst, label=int(edge_label[i]))

plt.figure(figsize=(12, 8))

pos = nx.spring_layout(G, seed=42)

nx.draw(
    G,
    pos,
    with_labels=True,
    node_size=500,
    font_size=8,
    arrows=True
)

plt.title("CICIDS Graph - First 100 Network Flows")
plt.show()