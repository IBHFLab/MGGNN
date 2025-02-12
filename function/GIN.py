import torch
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU


class GIN(torch.nn.Module):
    """Graph Isomorphism Network (GIN)"""

    def __init__(self, num_features, dim_h=64):
        """
        Initialize the GIN model.

        Args:
            num_features (int): The number of input features.
            dim_h (int): The hidden dimension size.
        """
        super(GIN, self).__init__()

        # GIN Conv layers
        self.conv1 = GINConv(
            Sequential(Linear(num_features, dim_h),
                       BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()))

        self.conv2 = GINConv(
            Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()))

        self.conv3 = GINConv(
            Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()))

        # Fully connected layers for graph-level classification
        self.lin1 = Linear(dim_h * 3, dim_h * 3)  # Concatenate 3 graph embeddings
        self.lin2 = Linear(dim_h * 3, 1)  # Output layer

    def forward(self, x, edge_index, batch):
        """
        Forward pass of the GIN model.

        Args:
            x (Tensor): Node feature matrix.
            edge_index (Tensor): Graph connectivity.
            batch (Tensor): Batch indices for pooling.

        Returns:
            Tensor: The output of the network after classification.
        """
        # Node embeddings through GIN Convolutional layers
        print(x.shape)
        h1 = self.conv1(x, edge_index)
        h2 = self.conv2(h1, edge_index)
        h3 = self.conv3(h2, edge_index)

        # Graph-level readout: pooling over the nodes to get a graph embedding
        h1 = global_add_pool(h1, batch)
        h2 = global_add_pool(h2, batch)
        h3 = global_add_pool(h3, batch)

        # Concatenate graph embeddings from all three layers
        h = torch.cat((h1, h2, h3), dim=1)

        # Classifier
        h = self.lin1(h)
        h = h.relu()
        h = F.dropout(h, p=0.5, training=self.training)
        print(h.shape)
        h = self.lin2(h)

        return h
