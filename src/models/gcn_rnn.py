from typing import Literal
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCNTemporalRNN(nn.Module):
    def __init__(self, in_feats, temporal_hidden, hidden_feats, out_feats, rnn_type="gru"):
        super().__init__()

        self.rnn_type = rnn_type

        # GRU layer for temporal modeling
        if self.rnn_type == "gru":
            self.rnn = nn.GRU(input_size=1, hidden_size=temporal_hidden, batch_first=True)
        else:
            self.rnn = nn.LSTM(input_size=1, hidden_size=temporal_hidden, batch_first=True)

        self.conv1 = GCNConv(temporal_hidden, hidden_feats)
        self.conv2 = GCNConv(hidden_feats, hidden_feats)
        self.fc = nn.Linear(hidden_feats, out_feats)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        edge_weight = edge_attr.squeeze() if edge_attr is not None else None

        x_seq = x.unsqueeze(-1)

        if self.rnn_type == "gru":
            out, h_n = self.rnn(x_seq)
        else:
            out, (h_n, c_n) = self.rnn(x_seq)

        x = out[:, -1, :]

        x = F.relu(self.conv1(x, edge_index, edge_weight=edge_weight))
        x = F.relu(self.conv2(x, edge_index, edge_weight=edge_weight))
        x = self.fc(x)
        return x