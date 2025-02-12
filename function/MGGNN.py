class MGGNN_model(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        edge_dim: int,
        num_layers: int,
        num_timesteps: int,
        slices: int,
        dropout: float = 0.0,
        brics=True,
    ):
        super().__init__()

        # Parameters initialization
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        self.num_layers = num_layers
        self.num_timesteps = num_timesteps
        self.dropout = dropout
        self.brics = brics
        self.slices = slices

        # Linear layers
        self.lin1 = Linear(in_channels, hidden_channels)
        self.lin_a = Linear(in_channels, hidden_channels)
        self.lin_b = Linear(edge_dim, hidden_channels)

        # Gate convolution layer
        self.gate_conv = GATEConv(hidden_channels, hidden_channels, edge_dim, dropout)
        self.gru = GRUCell(hidden_channels, hidden_channels)

        self.atom_convs = torch.nn.ModuleList()
        self.atom_grus = torch.nn.ModuleList()
        self.frag_convs = torch.nn.ModuleList()

        # Atom-based GAT layers
        for _ in range(num_layers - 1):
            conv = GATConv(hidden_channels, hidden_channels, dropout=dropout,
                           add_self_loops=False, negative_slope=0.01)
            self.atom_convs.append(conv)
            self.atom_grus.append(GRUCell(hidden_channels, hidden_channels))

        # Fragment-based NTN layers
        for _ in range(num_layers):
            conv = NTNConv(hidden_channels, hidden_channels, slices=slices,
                           dropout=dropout, edge_dim=hidden_channels)
            self.frag_convs.append(conv)

        self.lin_gate = Linear(3 * hidden_channels, hidden_channels)

        # Multi-view representation with BRICS
        if self.brics:
            self.cross_att = GATConv(hidden_channels, hidden_channels, heads=8,
                                     dropout=dropout, add_self_loops=False,
                                     negative_slope=0.01, concat=False)
            self.out = Linear(2 * hidden_channels, hidden_channels)
            self.out1 = Linear(2 * hidden_channels, 10)
            self.out2 = Linear(10, out_channels)

        # Molecule-level convolution and GRU
        self.mol_conv = GATConv(hidden_channels, hidden_channels,
                                dropout=dropout, add_self_loops=False,
                                negative_slope=0.01)
        self.mol_conv.explain = False
        self.mol_gru = GRUCell(hidden_channels, hidden_channels)

        self.lin2 = Linear(hidden_channels, out_channels)

        # Reset model parameters
        self.reset_parameters()

    def reset_parameters(self):
        """Resets all learnable parameters of the module."""
        self.lin1.reset_parameters()
        self.gate_conv.reset_parameters()
        self.gru.reset_parameters()
        for conv, gru in zip(self.atom_convs, self.atom_grus):
            conv.reset_parameters()
            gru.reset_parameters()
        self.mol_conv.reset_parameters()
        self.mol_gru.reset_parameters()
        self.lin2.reset_parameters()
        if self.brics:
            self.cross_att.reset_parameters()
            self.out.reset_parameters()
        for conv in self.frag_convs:
            conv.reset_parameters()

    def forward(self, data):
        """Forward pass for the MGGNN model."""
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        batch = data.batch

        x = F.leaky_relu_(self.lin1(x))
        h = F.elu_(self.gate_conv(x, edge_index, edge_attr))
        h = F.dropout(h, p=self.dropout, training=self.training)
        x = self.gru(h, x).relu_()

        # Atom-level convolutions and GRUs
        for conv, gru in zip(self.atom_convs, self.atom_grus):
            h = conv(x, edge_index)
            h = F.elu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = gru(h, x).relu()

        atom_x = global_add_pool(x, batch).relu_()

        # Molecule-level embedding
        row = torch.arange(batch.size(0), device=batch.device)
        edge_index = torch.stack([row, batch], dim=0)
        out = global_add_pool(x, batch).relu_()

        for t in range(self.num_timesteps):
            h = F.elu_(self.mol_conv((x, out), edge_index))
            h = F.dropout(h, p=self.dropout, training=self.training)
            out = self.mol_gru(h, out).relu_()

        if self.brics:
            mol_vec_ = global_add_pool(x, batch)
            mol_vec = global_add_pool(x, batch).relu_()

            # Fragment input and processing
            fra_x = data.x
            fra_edge_index = data.fra_edge_index
            fra_edge_attr = data.fra_edge_attr
            fra_x = F.relu(self.lin_a(fra_x))
            cluster = data.cluster_index
            fra_edge_attr = F.leaky_relu_(self.lin_b(fra_edge_attr))

            # Fragment convolution block
            for i in range(self.num_layers):
                fra_h = F.relu(self.frag_convs[i](fra_x, fra_edge_index, fra_edge_attr)[0])
                beta = self.lin_gate(torch.cat([fra_x, fra_h, fra_x - fra_h], 1)).sigmoid()
                fra_x = beta * fra_x + (1 - beta) * fra_h

            fra_x_ = global_add_pool(fra_x, cluster)
            fra_x = global_add_pool(fra_x, cluster).relu_()

            # Fragment clustering
            cluster, perm = consecutive_cluster(cluster)
            fra_batch = pool_batch(perm, data.batch)

            # Molecule-fragment attention
            row = torch.arange(fra_batch.size(0), device=batch.device)
            mol_fra_index = torch.stack([row, fra_batch], dim=0)
            fra_vec, att = self.cross_att((fra_x, mol_vec), mol_fra_index, return_attention_weights=True)
            fra_vec = fra_vec.relu()

            # Concatenating the results
            vectors_concat = [mol_vec, fra_vec]
            out_brics = torch.cat(vectors_concat, 1)

            mol_vec_ = mol_vec_[fra_batch]
            att = F.cosine_similarity(mol_vec_, fra_x_, dim=1), mol_fra_index

            out_brics = F.dropout(out_brics, p=self.dropout, training=self.training)
            out_brics = self.out(out_brics)

            vectors_concat_ = [out_brics, out]
            out = torch.cat(vectors_concat_, 1)

            emb = self.out1(out)
            out = self.out2(emb)

            return F.sigmoid(out)
