import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_add_pool


class GAT(torch.nn.Module):
    """Graph Attention Network"""

    def __init__(self, dim_in=92, dim_h=64, dim_out=1, heads=8):
        super().__init__()

        self.gat1 = GATv2Conv(dim_in, dim_h, heads=heads)
        self.gat2 = GATv2Conv(dim_h * heads, dim_out, heads=1)

    def forward(self, x, edge_index, batch):
        """
        Forward pass of the Graph Attention Network.
        """
        # Apply dropout and first GAT layer
        h = F.dropout(x, p=0.6, training=self.training)
        h = self.gat1(h, edge_index)
        h = F.elu(h)

        # Apply dropout and second GAT layer
        h = F.dropout(h, p=0.6, training=self.training)
        h = self.gat2(h, edge_index)

        # Global pooling (sum of node features in each graph)
        h = global_add_pool(h, batch)

        return h
