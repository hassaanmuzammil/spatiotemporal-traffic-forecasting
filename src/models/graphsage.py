import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.nn.norm import LayerNorm


class GraphSage(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, num_layers=3, dropout=0.3):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.hidden_feats = hidden_feats

        # Input layer
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.res_projs = nn.ModuleList()  # For residual projections

        # First layer
        self.convs.append(SAGEConv(in_feats, hidden_feats))
        self.norms.append(LayerNorm(hidden_feats))
        self.res_projs.append(nn.Linear(in_feats, hidden_feats))  # project input to hidden_feats

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_feats, hidden_feats))
            self.norms.append(LayerNorm(hidden_feats))
            self.res_projs.append(nn.Identity())  # hidden_feats → hidden_feats, identity

        # Output layer
        self.convs.append(SAGEConv(hidden_feats, hidden_feats))
        self.norms.append(LayerNorm(hidden_feats))
        self.res_projs.append(nn.Identity())

        # Final fully connected layer to predict horizon
        self.fc = nn.Linear(hidden_feats, out_feats)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        # SAGEConv does not support edge_weight
        # edge_weight = edge_attr.squeeze() if edge_attr is not None else None

        for i in range(self.num_layers):
            residual = self.res_projs[i](x)  # project residual to match hidden_feats
            x = self.convs[i](x, edge_index)
            x = self.norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + residual

        x = self.fc(x)
        return x