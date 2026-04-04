import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCNTemporalGRU(nn.Module):
    def __init__(self, in_feats, temporal_hidden, hidden_feats, out_feats):
        super().__init__()
        # GRU layer for temporal modeling
        self.rnn = nn.GRU(input_size=1, hidden_size=temporal_hidden, batch_first=True)

        self.conv1 = GCNConv(temporal_hidden, hidden_feats)
        self.conv2 = GCNConv(hidden_feats, hidden_feats)
        self.fc = nn.Linear(hidden_feats, out_feats)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        edge_weight = edge_attr.squeeze() if edge_attr is not None else None

        x_seq = x.unsqueeze(-1)  # [num_nodes, seq_len=12, features=1]  -> reshape to [num_nodes, seq_len=12, feature=1] for GRU
        out, h_n = self.rnn(x_seq)  # GRU input_size=1
        x = out[:, -1, :]           # last timestep embedding
        ###

        x = F.relu(self.conv1(x, edge_index, edge_weight=edge_weight))
        x = F.relu(self.conv2(x, edge_index, edge_weight=edge_weight))
        x = self.fc(x)
        return x