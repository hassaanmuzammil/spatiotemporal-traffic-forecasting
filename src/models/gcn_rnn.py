from typing import Literal
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCNTemporalRNN(nn.Module):
    def __init__(
        self,
        seq_len: int,
        temporal_hidden: int,
        hidden_feats: int,
        out_feats: int,
        rnn_type: Literal["gru", "lstm"] = "gru",
        input_dim: int = 1,
    ):
        super().__init__()

        # Number of historical timesteps in each input window.
        # Example: if using past 12 timesteps, seq_len = 12.
        self.seq_len = seq_len

        # Number of features per timestep.
        # Old format: 1  -> speed only
        # New format: 7 -> speed + 6 temporal features
        self.input_dim = input_dim

        self.rnn_type = rnn_type.lower()

        # Temporal encoder:
        # The RNN reads a sequence for each node and produces a hidden
        # representation summarizing the node's recent temporal behavior.
        if self.rnn_type == "gru":
            self.rnn = nn.GRU(
                input_size=self.input_dim,
                hidden_size=temporal_hidden,
                batch_first=True,
            )
        else:
            self.rnn = nn.LSTM(
                input_size=self.input_dim,
                hidden_size=temporal_hidden,
                batch_first=True,
            )

        # Graph convolutions:
        # After temporal encoding, apply graph convolution layers so each
        # node can incorporate information from neighboring sensors.
        self.conv1 = GCNConv(temporal_hidden, hidden_feats)
        self.conv2 = GCNConv(hidden_feats, hidden_feats)

        # Final projection to forecasting horizon.
        # Example: out_feats = 12 for predicting next 12 timesteps.
        self.fc = nn.Linear(hidden_feats, out_feats)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        edge_weight = edge_attr.squeeze() if edge_attr is not None else None

        num_nodes, num_features = x.shape

        # ------------------------------------------------------------
        # Backward compatibility handling
        # ------------------------------------------------------------
        # Case 1: Old dataset format
        #   x shape = [num_nodes, seq_len]
        #   Each timestep has only 1 feature (speed).
        #
        # Example:
        #   [207, 12] -> reshape to [207, 12, 1]
        #
        # Case 2: New dataset format with temporal features
        #   x shape = [num_nodes, seq_len * input_dim]
        #   Each timestep has multiple features:
        #   [speed, tod_sin, tod_cos, dow_sin, dow_cos, is_weekend, is_holiday]
        #
        # Example:
        #   [207, 84] with seq_len=12 and input_dim=7
        #   -> reshape to [207, 12, 7]
        # ------------------------------------------------------------
        if num_features == self.seq_len:
            x_seq = x.unsqueeze(-1)

        elif num_features == self.seq_len * self.input_dim:
            x_seq = x.view(num_nodes, self.seq_len, self.input_dim)

        else:
            raise ValueError(
                f"Unexpected input shape {x.shape}. "
                f"Expected second dimension to be either "
                f"{self.seq_len} (old format) or "
                f"{self.seq_len * self.input_dim} (new format)."
            )

        # Run temporal sequence modeling per node.
        if self.rnn_type == "gru":
            out, _ = self.rnn(x_seq)
        else:
            out, _ = self.rnn(x_seq)

        # Use the hidden representation from the last timestep.
        # This summarizes the node's full input history window.
        x = out[:, -1, :]

        # Spatial learning across graph neighbors.
        x = F.relu(self.conv1(x, edge_index, edge_weight=edge_weight))
        x = F.relu(self.conv2(x, edge_index, edge_weight=edge_weight))

        # Forecast future values.
        x = self.fc(x)
        return x