import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool as gmp, global_add_pool as gap
from torch_geometric.nn import Linear


class GCN(torch.nn.Module):
    def __init__(self, num_features, embedding_size):
        """
        Initialize the Graph Convolutional Network (GCN) model.

        Args:
            num_features (int): The number of input features.
            embedding_size (int): The size of the embedding space.
        """
        super(GCN, self).__init__()
        torch.manual_seed(42)

        # GCN layers
        self.initial_conv = GCNConv(num_features, embedding_size)
        self.conv1 = GCNConv(embedding_size, embedding_size)
        self.conv2 = GCNConv(embedding_size, embedding_size)
        self.conv3 = GCNConv(embedding_size, embedding_size)

        # Output layer
        self.out = Linear(embedding_size * 2, 1)  # Output layer after concatenating pooled features

    def forward(self, x, edge_index, batch_index):
        """
        Forward pass of the GCN model.

        Args:
            x (Tensor): Node feature matrix.
            edge_index (Tensor): Graph connectivity.
            batch_index (Tensor): Batch indices for pooling.

        Returns:
            out (Tensor): The output of the network after classification.
            hidden (Tensor): The hidden state representation after pooling.
        """
        # First Conv layer
        hidden = self.initial_conv(x, edge_index)
        hidden = F.tanh(hidden)

        # Other Conv layers
        hidden = self.conv1(hidden, edge_index)
        hidden = F.tanh(hidden)
        hidden = self.conv2(hidden, edge_index)
        hidden = F.tanh(hidden)
        hidden = self.conv3(hidden, edge_index)
        hidden = F.tanh(hidden)

        print(hidden.shape)
        print('++++++')

        # Global Pooling (stack different aggregations)
        hidden = torch.cat([gmp(hidden, batch_index), gap(hidden, batch_index)], dim=1)

        # Apply a final (linear) classifier.
        print(hidden.shape)
        out = self.out(hidden)

        return out, hidden
