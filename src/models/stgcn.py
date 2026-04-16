import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn.norm import LayerNorm

class TemporalBlock(nn.Module):
    """GRU-based temporal encoder — learns patterns across the input window."""
    def __init__(self, in_feats, hidden_feats, dropout):
        super().__init__()
        self.gru = nn.GRU(
            input_size = 1,
            hidden_size = hidden_feats,
            num_layers = 2,
            batch_first = True,
            dropout = dropout,
        )

    def forward(self, x):
        # x : [N, window_size]
        x = x.unsqueeze(-1) # [N, window_size, 1]
        _, h = self.gru(x) # h : [2, N, hidden]
        return h[-1] # [N, hidden]


class SpatialBlock(nn.Module):
    """GATv2 spatial layer — uses edge_index AND edge_attr as input to attention."""
    def __init__(self, hidden_feats, heads, dropout):
        super().__init__()
        assert hidden_feats % heads == 0, "hidden_feats must be divisible by heads"
        self.conv = GATv2Conv(
            in_channels = hidden_feats,
            out_channels = hidden_feats // heads,
            heads = heads,
            edge_dim = 1, # edge_attr has 1 feature (Gaussian weight)
            dropout = dropout,
            concat = True, # output = heads * (hidden // heads) = hidden
            add_self_loops = False, # already in adj_mx
        )
        self.norm = LayerNorm(hidden_feats)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr):
        residual = x
        x = self.conv(x, edge_index, edge_attr=edge_attr)
        x = self.norm(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x + residual


class SpatioTemporalGNN(nn.Module):
    """
    Architecture:
        1. TemporalBlock — GRU encodes time window → node embedding
        2. SpatialBlocks — stacked GATv2 layers propagate spatial context
        3. FC head — projects to forecast horizon

    Inputs (from PyG batch):
        data.x: [N, window_size]
        data.edge_index: [2, E]
        data.edge_attr: [E, 1]

    Output:
        [N, horizon]
    """
    def __init__(
        self,
        in_feats: int,
        hidden_feats: int,
        out_feats: int,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()

        # temporal encoder
        self.temporal = TemporalBlock(in_feats, hidden_feats, dropout)

        # spatial layers
        self.spatial = nn.ModuleList([
            SpatialBlock(hidden_feats, heads, dropout)
            for _ in range(num_layers)
        ])

        # output
        self.fc = nn.Sequential(
            nn.Linear(hidden_feats, hidden_feats),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_feats, out_feats),
        )

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # 1. temporal encoding
        x = self.temporal(x) # [N, hidden]

        # 2. spatial propagation
        for layer in self.spatial:
            x = layer(x, edge_index, edge_attr) # [N, hidden]

        # 3. predict
        return self.fc(x) # [N, horizon]