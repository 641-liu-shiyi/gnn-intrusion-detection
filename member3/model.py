"""E-GraphSAGE for edge classification (Lo et al., 2022), PyG 2.8."""
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax


class EdgeSAGEConv(MessagePassing):
    """h_v' = W_apply [h_v || AGG_u W_msg [h_u || e_uv]]  (edge features enter every message).

    aggr="mean":      AGG = plain average of incoming messages.
    aggr="attention": AGG = sum_u alpha_uv * m_uv, alpha = softmax over v's incoming edges of
                      LeakyReLU(a^T [W_q h_v || m_uv])  (GAT-style scoring, Velickovic et al. 2018).
    """

    def __init__(self, node_dim, edge_dim, out_dim, aggr="mean"):
        attention = aggr == "attention"
        super().__init__(aggr="add" if attention else aggr)
        self.attention = attention
        self.lin_msg = nn.Linear(node_dim + edge_dim, out_dim)
        self.lin_apply = nn.Linear(node_dim + out_dim, out_dim)
        if attention:
            # ponytail: single head; multi-head (concat/average) if attention runs prove unstable.
            self.lin_query = nn.Linear(node_dim, out_dim)
            self.att = nn.Linear(2 * out_dim, 1, bias=False)

    def forward(self, x, edge_index, edge_attr):
        agg = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return self.lin_apply(torch.cat([x, agg], dim=-1))

    def message(self, x_i, x_j, edge_attr, index, ptr, size_i):
        m = self.lin_msg(torch.cat([x_j, edge_attr], dim=-1))
        if not self.attention:
            return m
        score = F.leaky_relu(self.att(torch.cat([self.lin_query(x_i), m], dim=-1)), 0.2)
        return softmax(score, index, ptr, size_i) * m


class EGraphSAGE(nn.Module):
    """aggr: "mean" | "sum" | "attention".  Ablation switches (defaults = baseline):
    head_edge_attr: classifier also sees the flow's own features, not just its two hosts.
    bidirectional:  messages also flow dst->src (with a direction flag column).
    num_layers=0 with head_edge_attr=True removes the graph entirely (flow features only)."""

    def __init__(self, edge_dim, hidden_dim, num_layers, num_classes, dropout=0.2, aggr="mean",
                 head_edge_attr=False, bidirectional=True):
        super().__init__()
        self.head_edge_attr = head_edge_attr
        self.bidirectional = bidirectional
        msg_edge_dim = edge_dim + 1 if bidirectional else edge_dim
        dims = [1] + [hidden_dim] * num_layers  # node input = constant 1
        self.convs = nn.ModuleList(
            EdgeSAGEConv(dims[i], msg_edge_dim, dims[i + 1], aggr) for i in range(num_layers)
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(2 * dims[-1] + (edge_dim if head_edge_attr else 0), num_classes)

    def forward(self, x, edge_index, edge_attr):
        if self.bidirectional:
            # A host that only sends (e.g. a scanner) still aggregates its flows;
            # the flag column keeps forward vs reverse copies distinguishable.
            n_e = edge_attr.size(0)
            flag = torch.cat([edge_attr.new_zeros(n_e, 1), edge_attr.new_ones(n_e, 1)])
            ei = torch.cat([edge_index, edge_index.flip(0)], dim=1)
            ea = torch.cat([torch.cat([edge_attr, edge_attr]), flag], dim=1)
        else:
            ei, ea = edge_index, edge_attr

        for conv in self.convs:
            x = self.dropout(torch.relu(conv(x, ei, ea)))

        src, dst = edge_index
        z = [x[src], x[dst]] + ([edge_attr] if self.head_edge_attr else [])
        return self.head(torch.cat(z, dim=-1))
