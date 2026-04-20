import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv


class GCNSpatioTemporalAttention(nn.Module):
    """
    Spatio-temporal model with temporal self-attention followed by graph attention.

    Supports both input formats:

    1. Old format:
        data.x -> [N, window_size]
        Example: [207, 12]
        Meaning: one feature per timestep (speed only)

    2. New format:
        data.x -> [N, window_size * in_feats]
        Example: [207, 84]
        Meaning: multiple features per timestep
                 [speed,
                  time_of_day_sin, time_of_day_cos,
                  day_of_week_sin, day_of_week_cos,
                  is_weekend, is_holiday]
    """
    def __init__(
        self,
        in_feats=1,
        temporal_hidden=32,
        hidden_feats=32,
        out_feats=1,
        num_heads=4,
        window_size=12,
    ):
        super().__init__()

        self.in_feats = in_feats
        self.window_size = window_size

        # Projection for new multifeature input per timestep
        self.input_proj = nn.Linear(in_feats, temporal_hidden)

        # Projection kept for old speed-only input per timestep
        self.old_input_proj = nn.Linear(1, temporal_hidden)

        # Transformer-style temporal attention across timesteps
        self.attn = nn.MultiheadAttention(
            embed_dim=temporal_hidden,
            num_heads=num_heads,
            batch_first=True,
        )

        self.norm = nn.LayerNorm(temporal_hidden)

        # Spatial graph layers
        # self.conv1 = GCNConv(temporal_hidden, hidden_feats)
        # self.conv2 = GCNConv(hidden_feats, hidden_feats)
        self.conv1 = GATConv(temporal_hidden, hidden_feats, num_heads, concat=False)
        self.conv2 = GATConv(hidden_feats, hidden_feats, num_heads, concat=False)

        self.fc = nn.Linear(hidden_feats, out_feats)

    def reshape_temporal_input(self, x):
        """
        Convert input into sequence format for temporal attention.

        Supported cases:
        - Old: [N, window_size] -> [N, window_size, 1]
        - New: [N, window_size * in_feats] -> [N, window_size, in_feats]
        - Already 3D: returned as-is
        """
        if x.dim() == 3:
            return x

        if x.dim() == 2 and x.size(1) == self.window_size:
            return x.unsqueeze(-1)

        expected_flattened_width = self.window_size * self.in_feats
        if x.dim() == 2 and x.size(1) == expected_flattened_width:
            return x.view(x.size(0), self.window_size, self.in_feats)

        raise ValueError(
            f"Unexpected input shape {tuple(x.shape)}. "
            f"Expected [N, {self.window_size}] or [N, {expected_flattened_width}]."
        )

    def project_temporal_input(self, x_seq):
        """
        Project timestep features into temporal attention embedding space.
        """
        if x_seq.size(-1) == 1:
            return self.old_input_proj(x_seq)

        if x_seq.size(-1) == self.in_feats:
            return self.input_proj(x_seq)

        raise ValueError(
            f"Unexpected temporal feature dimension {x_seq.size(-1)}. "
            f"Expected 1 or {self.in_feats}."
        )

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # Keep current edge handling behavior
        edge_weight = edge_attr.squeeze() if edge_attr is not None else None

        # Step 1: reshape input into [N, window_size, features_per_timestep]
        x_seq = self.reshape_temporal_input(x)

        # Step 2: project timestep features to temporal_hidden
        x_seq = self.project_temporal_input(x_seq)

        # Step 3: self-attention over timesteps
        attn_out, _ = self.attn(x_seq, x_seq, x_seq)
        x_seq = self.norm(attn_out + x_seq)

        # Step 4: temporal pooling to get one embedding per node
        x = x_seq.mean(dim=1)

        # Step 5: spatial propagation
        # x = F.relu(self.conv1(x, edge_index, edge_weight=edge_weight))
        # x = F.relu(self.conv2(x, edge_index, edge_weight=edge_weight))
        x = F.relu(self.conv1(x, edge_index, edge_attr=edge_weight))
        x = F.relu(self.conv2(x, edge_index, edge_attr=edge_weight))

        # Step 6: final prediction
        x = self.fc(x)

        return x