import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn.norm import LayerNorm


class TemporalBlock(nn.Module):
    """
    GRU-based temporal encoder.

    Purpose:
        Encodes the input time window for each sensor/node into a single hidden embedding.

    Supported input shapes:
        1. Old format  -> [N, window_size]
           - only one feature per timestep (speed)

        2. New format  -> [N, window_size, in_feats]
           - multiple features per timestep
           - example:
                [speed,
                 time_of_day_sin, time_of_day_cos,
                 day_of_week_sin, day_of_week_cos,
                 is_weekend, is_holiday]

    Output shape:
        [N, hidden_feats]
    """
    def __init__(self, in_feats, hidden_feats, dropout):
        super().__init__()

        self.in_feats = in_feats
        self.hidden_feats = hidden_feats

        # Main GRU for the new multifeature temporal input.
        self.gru = nn.GRU(
            input_size=in_feats,
            hidden_size=hidden_feats,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )

        # Backward-compatible GRU for the old single-feature input.
        # This allows old data.x = [N, window_size] to continue working
        # without forcing the trainer or dataset to change.
        self.single_feature_gru = nn.GRU(
            input_size=1,
            hidden_size=hidden_feats,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )

    def forward(self, x):
        """
        Args:
            x:
                - old format: [N, window_size]
                - new format: [N, window_size, in_feats]

        Returns:
            Tensor of shape [N, hidden_feats]
        """
        # Old format: only historical speed values are present.
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # [N, window_size, 1]
            _, h = self.single_feature_gru(x)
            return h[-1]

        # New format: multiple features exist at each timestep.
        if x.dim() == 3:
            _, h = self.gru(x)
            return h[-1]

        raise ValueError(
            f"TemporalBlock received unexpected input shape {tuple(x.shape)}. "
            "Expected [N, window_size] or [N, window_size, in_feats]."
        )


class SpatialBlock(nn.Module):
    """
    Graph attention block for spatial message passing across the road network.

    Purpose:
        Once each node has a temporal embedding, this layer lets nodes exchange
        information with neighboring nodes using graph attention.

    Inputs:
        x          -> [N, hidden_feats]
        edge_index -> [2, E]
        edge_attr  -> [E, 1]

    Output:
        [N, hidden_feats]
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


class SpatioTemporalGNN(nn.Module):
    """
    Spatio-temporal traffic forecasting model.

    Backward-compatible input support:

    1. Old pipeline:
        data.x -> [N, window_size]
        Example:
            [207, 12]

    2. New pipeline with temporal features:
        data.x -> [N, window_size * in_feats]
        Example:
            window_size = 12
            in_feats = 7
            flattened width = 84
            so data.x -> [207, 84]

    Architecture:
        1. TemporalBlock:
            Encodes the historical input window for each node with a GRU.

        2. SpatialBlock(s):
            Propagate information across connected sensors using graph attention.

        3. Fully connected prediction head:
            Maps the final node embedding to the forecast horizon.
    """
    def __init__(
        self,
        in_feats: int,
        hidden_feats: int,
        out_feats: int,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.3,
        window_size: int = 12,
    ):
        super().__init__()

        # Number of features per timestep for the new temporal-feature input.
        # Example:
        #   speed,
        #   time_of_day_sin, time_of_day_cos,
        #   day_of_week_sin, day_of_week_cos,
        #   is_weekend, is_holiday
        self.in_feats = in_feats

        # Number of historical timesteps used as input.
        self.window_size = window_size

        # 1. Temporal encoder
        self.temporal = TemporalBlock(in_feats, hidden_feats, dropout)

        # 2. Spatial propagation layers
        self.spatial = nn.ModuleList([
            SpatialBlock(hidden_feats, heads, dropout)
            for _ in range(num_layers)
        ])

        # 3. Prediction head
        self.fc = nn.Sequential(
            nn.Linear(hidden_feats, hidden_feats),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_feats, out_feats),
        )

    def reshape_temporal_input(self, x):
        """
        Convert incoming node features into the format expected by TemporalBlock.

        Supported cases:

        Case 1: Old dataset format
            x -> [N, window_size]
            Return as-is.
            TemporalBlock will treat it as one feature per timestep.

        Case 2: New flattened temporal-feature format
            x -> [N, window_size * in_feats]
            Reshape to:
                [N, window_size, in_feats]

        Case 3: Already reshaped input
            x -> [N, window_size, in_feats]
            Return as-is.
        """
        # Already in sequence form
        if x.dim() == 3:
            return x

        # Old format: only speed history
        if x.dim() == 2 and x.size(1) == self.window_size:
            return x

        # New flattened format: reshape back into [N, window_size, in_feats]
        expected_flattened_width = self.window_size * self.in_feats
        if x.dim() == 2 and x.size(1) == expected_flattened_width:
            return x.view(x.size(0), self.window_size, self.in_feats)

        raise ValueError(
            f"Unexpected input shape {tuple(x.shape)} for SpatioTemporalGNN. "
            f"Expected [N, {self.window_size}] for old format or "
            f"[N, {expected_flattened_width}] for flattened temporal format."
        )

    def forward(self, data):
        """
        Forward pass.

        Args:
            data: PyG batch object containing
                - data.x
                - data.edge_index
                - data.edge_attr

        Returns:
            Forecast tensor of shape [N, horizon]
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # Convert the incoming node features into the correct temporal format.
        x = self.reshape_temporal_input(x)

        # Temporal encoding:
        # - old format  -> GRU sees 1 feature per timestep
        # - new format  -> GRU sees multiple features per timestep
        x = self.temporal(x)

        # Spatial propagation across graph neighbors
        for layer in self.spatial:
            x = layer(x, edge_index, edge_attr)

        # Predict future traffic values over the forecast horizon
        return self.fc(x)