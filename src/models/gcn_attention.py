import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv

class GCNSpatioTemporalAttention(nn.Module):
    def __init__(self, in_feats=1, temporal_hidden=32, hidden_feats=32, out_feats=1, num_heads=4):
        super().__init__()
        # project scalar timestep → temporal_hidden so MultiheadAttention has proper embedding
        self.input_proj = nn.Linear(in_feats, temporal_hidden)

        # transformer-style attention over the 12 timesteps
        self.attn = nn.MultiheadAttention(
            embed_dim=temporal_hidden,
            num_heads=num_heads,
            batch_first=True,  # keeps [N, seq_len, embed_dim] format
        )
        self.norm = nn.LayerNorm(temporal_hidden)

        # Conv layers for spatial aggregation
        # GCNConv: fixed averaging
        # GATConv: learned, adaptive weighting

        # self.conv1 = GCNConv(temporal_hidden, hidden_feats)
        # self.conv2 = GCNConv(hidden_feats, hidden_feats)
        self.conv1 = GATConv(temporal_hidden, hidden_feats, num_heads, concat=False)
        self.conv2 = GATConv(hidden_feats, hidden_feats, num_heads, concat=False)   

        self.fc = nn.Linear(hidden_feats, out_feats)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        edge_weight = edge_attr.squeeze() if edge_attr is not None else None

        # x: [num_nodes, seq_len] → [num_nodes, seq_len, 1] for input projection
        x_seq = self.input_proj(x)
        x_seq = self.input_proj(x_seq)  # [num_nodes, seq_len, temporal_hidden]

        # self-attention over timesteps
        attn_out, _ = self.attn(x_seq, x_seq, x_seq)  # [num_nodes, seq_len, temporal_hidden]
        x_seq = self.norm(attn_out + x_seq)

        x = x_seq.mean(dim=1) # use temporal pooling instead

        # x = F.relu(self.conv1(x, edge_index, edge_weight=edge_weight))
        # x = F.relu(self.conv2(x, edge_index, edge_weight=edge_weight))
        x = F.relu(self.conv1(x, edge_index, edge_attr=edge_weight))
        x = F.relu(self.conv2(x, edge_index, edge_attr=edge_weight))

        x = self.fc(x)

        return x