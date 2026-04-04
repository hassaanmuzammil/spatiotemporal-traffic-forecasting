
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

# Simple GCN Baseline
class GCNBaseline(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats):
        super().__init__()
        self.conv1 = GCNConv(in_feats, hidden_feats)
        self.conv2 = GCNConv(hidden_feats, hidden_feats)
        self.fc = nn.Linear(hidden_feats, out_feats)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # edge_attr must be shape [num_edges]
        # edge_weight = None
        edge_weight = edge_attr.squeeze() if edge_attr is not None else None

        x = F.relu(self.conv1(x, edge_index, edge_weight=edge_weight))
        x = F.relu(self.conv2(x, edge_index, edge_weight=edge_weight))
        x = self.fc(x)
        return x
    

if __name__ == "__main__":

    from src.configs.config import (
        in_feats, 
        hidden_feats, 
        out_feats, 
        device,
    )
    model = GCNBaseline(in_feats, hidden_feats, out_feats).to(device)
