import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj

class DiffusionConv(nn.Module):
    def __init__(self, hidden_feats):
        super().__init__()
        self.linear = nn.Linear(hidden_feats, hidden_feats)

    def forward(self, x, adj):
        return self.linear(torch.matmul(adj, x))


class DCGRUCell(nn.Module):
    def __init__(self, hidden_feats):
        super().__init__()
        self.hidden_feats = hidden_feats

        self.conv_z = DiffusionConv(hidden_feats)
        self.conv_r = DiffusionConv(hidden_feats)
        self.conv_h = DiffusionConv(hidden_feats)

    def forward(self, x, h, adj):
        z = torch.sigmoid(self.conv_z(x, adj) + self.conv_z(h, adj))
        r = torch.sigmoid(self.conv_r(x, adj) + self.conv_r(h, adj))
        h_tilde = torch.tanh(self.conv_h(x, adj) + self.conv_h(r * h, adj))
        return (1 - z) * h + z * h_tilde


class DCRNN(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, num_layers=2, dropout=0.3):
        super().__init__()
        self.hidden_feats = hidden_feats
        self.num_layers = num_layers

        self.input_proj = nn.Linear(1, hidden_feats)
        self.cells = nn.ModuleList([DCGRUCell(hidden_feats) for _ in range(num_layers)])

        self.fc = nn.Sequential(
            nn.Linear(hidden_feats, hidden_feats),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_feats, out_feats),
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # build dense adjacency
        adj = to_dense_adj(edge_index)[0]  # [N, N]

        x = x.unsqueeze(-1)  # [N, T, 1]
        x = self.input_proj(x)

        h = [torch.zeros(x.size(0), self.hidden_feats, device=x.device)
             for _ in range(self.num_layers)]

        for t in range(x.size(1)):
            inp = x[:, t, :]
            for i, cell in enumerate(self.cells):
                h[i] = cell(inp, h[i], adj)
                inp = h[i]

        return self.fc(h[-1])