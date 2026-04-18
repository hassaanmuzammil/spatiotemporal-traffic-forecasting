import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn.norm import LayerNorm


class TemporalTransformerBlock(nn.Module):
    """
    Transformer-based temporal encoder.
    Replaces the GRU in SpatioTemporalGNN with multi-layer self-attention
    to capture long-range dependencies across the input window.
    """
    def __init__(self, hidden_feats, num_heads, num_layers, dropout):
        super().__init__()

        # project each scalar speed value → hidden_feats embedding
        self.input_proj = nn.Linear(1, hidden_feats)

        # positional encoding — tells the model which timestep is which
        self.pos_embedding = nn.Embedding(512, hidden_feats)

        # stack of Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_feats,
            nhead=num_heads,
            dim_feedforward=hidden_feats * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.norm = nn.LayerNorm(hidden_feats)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [N, window_size]
        N, T = x.shape

        # expand scalar → embedding
        x = x.unsqueeze(-1)                          # [N, T, 1]
        x = self.input_proj(x)                       # [N, T, hidden_feats]

        # add positional encoding so model knows timestep order
        positions = torch.arange(T, device=x.device)
        x = x + self.pos_embedding(positions)        # [N, T, hidden_feats]

        # run through transformer encoder
        x = self.transformer(x)                      # [N, T, hidden_feats]
        x = self.norm(x)

        # mean pool across all timesteps → one vector per sensor
        x = x.mean(dim=1)                            # [N, hidden_feats]
        return self.dropout(x)


class SpatialBlock(nn.Module):
    """
    GATv2 spatial layer — identical to stgcn.py.
    Kept the same so results are directly comparable.
    """
    def __init__(self, hidden_feats, heads, dropout):
        super().__init__()
        assert hidden_feats % heads == 0, "hidden_feats must be divisible by heads"
        self.conv = GATv2Conv(
            in_channels=hidden_feats,
            out_channels=hidden_feats // heads,
            heads=heads,
            edge_dim=1,
            dropout=dropout,
            concat=True,
            add_self_loops=False,
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


class SpatioTemporalTransformer(nn.Module):
    """
    Transformer-based spatio-temporal GNN for traffic forecasting.

    Architecture:
        1. TemporalTransformerBlock — multi-layer self-attention encodes time window
        2. SpatialBlocks            — stacked GATv2 layers propagate spatial context
        3. FC head                  — projects to forecast horizon

    Inputs (from PyG batch):
        data.x          : [N, window_size]
        data.edge_index : [2, E]
        data.edge_attr  : [E, 1]

    Output:
        [N, horizon]
    """
    def __init__(
        self,
        in_feats: int,
        hidden_feats: int,
        out_feats: int,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.3,
        temporal_layers: int = 2,
    ):
        super().__init__()

        # temporal encoder (Transformer)
        self.temporal = TemporalTransformerBlock(
            hidden_feats=hidden_feats,
            num_heads=num_heads,
            num_layers=temporal_layers,
            dropout=dropout,
        )

        # spatial layers (same as stgcn.py for fair comparison)
        self.spatial = nn.ModuleList([
            SpatialBlock(hidden_feats, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # output head
        self.fc = nn.Sequential(
            nn.Linear(hidden_feats, hidden_feats),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_feats, out_feats),
        )

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # 1. temporal encoding (Transformer)
        x = self.temporal(x)                         # [N, hidden]

        # 2. spatial propagation (GATv2)
        for layer in self.spatial:
            x = layer(x, edge_index, edge_attr)      # [N, hidden]

        # 3. predict
        return self.fc(x)                            # [N, horizon]