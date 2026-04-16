import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class DilatedTemporalConv(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.conv = nn.Conv1d(
            channels, channels,
            kernel_size=2,
            dilation=dilation,
            padding=dilation
        )

    def forward(self, x):
        out = self.conv(x)
        return out[:, :, :x.size(2)]  # trim back to original T


class GraphWaveNetLayer(nn.Module):
    def __init__(self, hidden_feats, dropout):
        super().__init__()
        self.filter_conv = DilatedTemporalConv(hidden_feats, dilation=1)
        self.gate_conv   = DilatedTemporalConv(hidden_feats, dilation=1)

        self.gcn = GCNConv(hidden_feats, hidden_feats)
        self.dropout = dropout

    def forward(self, x, edge_index):
        # x: [N, H, T]
        residual = x

        filter_out = torch.tanh(self.filter_conv(x))
        gate_out   = torch.sigmoid(self.gate_conv(x))
        x = filter_out * gate_out

        x = x.transpose(1, 2)  # [N, T, H]

        out = []
        for t in range(x.size(1)):
            out.append(self.gcn(x[:, t, :], edge_index))
        x = torch.stack(out, dim=1)

        x = x.transpose(1, 2)
        x = F.dropout(x, p=self.dropout, training=self.training)

        return x + residual


class GraphWaveNet(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, num_layers=3, dropout=0.3):
        super().__init__()

        self.input_proj = nn.Linear(1, hidden_feats)

        self.layers = nn.ModuleList([
            GraphWaveNetLayer(hidden_feats, dropout)
            for _ in range(num_layers)
        ])

        self.fc = nn.Sequential(
            nn.Linear(hidden_feats, hidden_feats),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_feats, out_feats),
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = x.unsqueeze(-1)          # [N, T, 1]
        x = self.input_proj(x)       # [N, T, H]
        x = x.transpose(1, 2)        # [N, H, T]

        for layer in self.layers:
            x = layer(x, edge_index)

        x = x[:, :, -1]              # last timestep
        return self.fc(x)