import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
from tqdm import tqdm
import sklearn
import torch
import matplotlib

from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.data import DataLoader
import rdkit
from rdkit import Chem
from rdkit.Chem.BRICS import FindBRICSBonds
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger
#########################################################################################
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Linear, Sequential, Parameter, Bilinear
from sklearn.metrics import roc_auc_score
from torch_scatter import scatter
from torch_geometric.nn import global_add_pool, GATConv,AttentiveFP
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, reset
from torch_geometric.nn.pool.pool import pool_batch
from torch_geometric.nn.pool.consecutive import consecutive_cluster

# %%
# ---------------------------------------
# Attention layers
# ---------------------------------------
class FeatureAttention(nn.Module):
    def __init__(self, channels, reduction):
        super().__init__()
        self.mlp = Sequential(
            Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            Linear(channels // reduction, channels, bias=False),
        )

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.mlp)

    def forward(self, x, batch, size=None):
        max_result = scatter(x, batch, dim=0, dim_size=size, reduce='max')
        sum_result = scatter(x, batch, dim=0, dim_size=size, reduce='sum')
        max_out = self.mlp(max_result)
        sum_out = self.mlp(sum_result)
        y = torch.sigmoid(max_out + sum_out)
        y = y[batch]
        return x * y


# %%
# ---------------------------------------
# Neural tensor networks conv
# ---------------------------------------
class NTNConv(MessagePassing):

    def __init__(self, in_channels, out_channels, slices, dropout, edge_dim=None, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(NTNConv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.slices = slices
        self.dropout = dropout
        self.edge_dim = edge_dim

        self.weight_node = Parameter(torch.Tensor(in_channels,
                                                  out_channels))
        if edge_dim is not None:
            self.weight_edge = Parameter(torch.Tensor(edge_dim,
                                                      out_channels))
        else:
            self.weight_edge = self.register_parameter('weight_edge', None)

        self.bilinear = Bilinear(out_channels, out_channels, slices, bias=False)

        if self.edge_dim is not None:
            self.linear = Linear(3 * out_channels, slices)
        else:
            self.linear = Linear(2 * out_channels, slices)

        self._alpha = None

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight_node)
        glorot(self.weight_edge)
        self.bilinear.reset_parameters()
        self.linear.reset_parameters()

    def forward(self, x, edge_index, edge_attr=True, return_attention_weights=None):

        x = torch.matmul(x, self.weight_node)

        if self.weight_edge is not None:
            assert edge_attr is not None
            edge_attr = torch.matmul(edge_attr, self.weight_edge)

        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)

        alpha = self._alpha
        self._alpha = None

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            return out, (edge_index, alpha)
        else:
            return out

    def message(self, x_i, x_j, edge_attr):
        score = self.bilinear(x_i, x_j)
        if edge_attr is not None:
            vec = torch.cat((x_i, edge_attr, x_j), 1)
            block_score = self.linear(vec)  # bias already included
        else:
            vec = torch.cat((x_i, x_j), 1)
            block_score = self.linear(vec)
        scores = score + block_score
        alpha = torch.tanh(scores)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        dim_split = self.out_channels // self.slices
        out = x_j.view(-1, self.slices, dim_split)

        out = out * alpha.view(-1, self.slices, 1)
        out = out.view(-1, self.out_channels)
        return out

    def __repr__(self):
        return '{}({}, {}, slices={})'.format(self.__class__.__name__,
                                              self.in_channels,
                                              self.out_channels, self.slices)


# %%
# ---------------------------------------
# HiGNN backbone
# ---------------------------------------
class HiGNN(torch.nn.Module):
    """Hierarchical informative graph neural network for molecular representation.

    """

    def __init__(self, in_channels, hidden_channels, out_channels, edge_dim, num_layers,
                 slices, dropout, f_att=False, r=4, brics=True, cl=False):
        super(HiGNN, self).__init__()

        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout

        self.f_att = f_att
        self.brics = brics
        self.cl = cl

        # atom feature transformation
        self.lin_a = Linear(in_channels, hidden_channels)
        self.lin_b = Linear(edge_dim, hidden_channels)

        # convs block
        self.atom_convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = NTNConv(hidden_channels, hidden_channels, slices=slices,
                           dropout=dropout, edge_dim=hidden_channels)
            self.atom_convs.append(conv)

        self.lin_gate = Linear(3 * hidden_channels, hidden_channels)

        if self.f_att:
            self.feature_att = FeatureAttention(channels=hidden_channels, reduction=r)

        if self.brics:
            # mol-fra attention
            self.cross_att = GATConv(hidden_channels, hidden_channels, heads=4,
                                     dropout=dropout, add_self_loops=False,
                                     negative_slope=0.01, concat=False)

        if self.brics:
            self.out = Linear(2 * hidden_channels, 10)
            self.out1 = Linear(10, out_channels)
        else:
            self.out = Linear(hidden_channels, out_channels)


        if self.cl:
            self.lin_project = Linear(hidden_channels, int(hidden_channels / 2))

        self.reset_parameters()

    def reset_parameters(self):

        self.lin_a.reset_parameters()
        self.lin_b.reset_parameters()

        for conv in self.atom_convs:
            conv.reset_parameters()

        self.lin_gate.reset_parameters()

        if self.f_att:
            self.feature_att.reset_parameters()

        if self.brics:
            self.cross_att.reset_parameters()

        self.out.reset_parameters()

        if self.cl:
            self.lin_project.reset_parameters()

    def forward(self, data):
        # get mol input
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        batch = data.batch

        x = F.relu(self.lin_a(x))  # (N, 46) -> (N, hidden_channels)
        edge_attr = F.relu(self.lin_b(edge_attr))  # (N, 10) -> (N, hidden_channels)

        # mol conv block
        for i in range(0, self.num_layers):
            h = F.relu(self.atom_convs[i](x, edge_index, edge_attr))
            beta = self.lin_gate(torch.cat([x, h, x - h], 1)).sigmoid()
            x = beta * x + (1 - beta) * h
            if self.f_att:
                x = self.feature_att(x, batch)

        mol_vec_ = global_add_pool(x, batch)
        mol_vec = global_add_pool(x, batch).relu_()

        if self.brics:
            # get fragment input
            fra_x = data.x
            fra_edge_index = data.fra_edge_index
            fra_edge_attr = data.fra_edge_attr
            cluster = data.cluster_index

            fra_x = F.relu(self.lin_a(fra_x))  # (N, 46) -> (N, hidden_channels)
            fra_edge_attr = F.leaky_relu_(self.lin_b(fra_edge_attr))  # (N, 10) -> (N, hidden_channels)

            # fragment convs block
            for i in range(0, self.num_layers):

                fra_h = F.relu(self.atom_convs[i](fra_x, fra_edge_index, fra_edge_attr))
                beta = self.lin_gate(torch.cat([fra_x, fra_h, fra_x - fra_h], 1)).sigmoid()
                fra_x = beta * fra_x + (1 - beta) * fra_h
                if self.f_att:
                    fra_x = self.feature_att(fra_x, cluster)

            fra_x_ = global_add_pool(fra_x, cluster)
            fra_x = global_add_pool(fra_x, cluster).relu_()

            # get fragment batch
            cluster, perm = consecutive_cluster(cluster)
            fra_batch = pool_batch(perm, data.batch)

            # molecule-fragment attention
            row = torch.arange(fra_batch.size(0), device=batch.device)
            mol_fra_index = torch.stack([row, fra_batch], dim=0)
            fra_vec, _ = self.cross_att((fra_x, mol_vec), mol_fra_index, return_attention_weights=False)
            fra_vec = fra_vec.relu()

            vectors_concat = list()
            vectors_concat.append(mol_vec)
            vectors_concat.append(fra_vec)

            out = torch.cat(vectors_concat, 1)

            mol_vec_ = mol_vec_[fra_batch]
            att = F.cosine_similarity(mol_vec_, fra_x_, dim=1), mol_fra_index

            # molecule-fragment contrastive
            if self.cl:
                out = F.dropout(out, p=self.dropout, training=self.training)
                return self.out(out), self.lin_project(mol_vec).relu_(), self.lin_project(fra_vec).relu_()
            else:
                out = F.dropout(out, p=self.dropout, training=self.training)
                out=self.out(out)
                emb=out
                out=self.out1(out)
                return F.sigmoid(out), emb

        else:
            assert self.cl is False
            out = F.dropout(mol_vec, p=self.dropout, training=self.training)
            emb = out
            out = self.out1(out)
            return F.sigmoid(out), emb
###################################################################################
class attentive_fp(torch.nn.Module):
    def __init__(self):
     super(attentive_fp,self).__init__()
     self.atfp1 = AttentiveFP(in_channels=46, hidden_channels=128,
                              out_channels=10, edge_dim=10,
                              num_layers=4, num_timesteps=7,
                             )


     self.out = Linear(10,1)
    def forward(self,data1):
        x, edge_index, batch, edge_attr = data1.x, data1.edge_index, data1.batch, data1.edge_attr

        x=F.dropout(x, p=0.6, training=self.training)

        h = self.atfp1(x, edge_index, edge_attr, batch)
        emd=h
        h=self.out(h)
        return F.sigmoid(h)



########################################################################

def onehot_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return [x == s for s in allowable_set]


def onehot_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]


def atom_attr(mol, explicit_H=False, use_chirality=True, pharmaco=True, scaffold=True):
    if pharmaco:
        mol = tag_pharmacophore(mol)
    if scaffold:
        mol = tag_scaffold(mol)

    feat = []
    for i, atom in enumerate(mol.GetAtoms()):
        results = onehot_encoding_unk(
            atom.GetSymbol(),
            ['B', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'As', 'Se', 'Br', 'Te', 'I', 'At', 'other'
             ]) + onehot_encoding_unk(atom.GetDegree(),
                                      [0, 1, 2, 3, 4, 5, 'other']) + \
                  [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
                  onehot_encoding_unk(atom.GetHybridization(), [
                      Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                      Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
                      Chem.rdchem.HybridizationType.SP3D2, 'other'
                  ]) + [atom.GetIsAromatic()]
        if not explicit_H:
            results = results + onehot_encoding_unk(atom.GetTotalNumHs(),
                                                    [0, 1, 2, 3, 4])
        if use_chirality:
            try:
                results = results + onehot_encoding_unk(
                    atom.GetProp('_CIPCode'),
                    ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
            # print(one_of_k_encoding_unk(atom.GetProp('_CIPCode'), ['R', 'S']) + [atom.HasProp('_ChiralityPossible')])
            except:
                results = results + [0, 0] + [atom.HasProp('_ChiralityPossible')]
        if pharmaco:
            results = results + [int(atom.GetProp('Hbond_donor'))] + [int(atom.GetProp('Hbond_acceptor'))] + \
                      [int(atom.GetProp('Basic'))] + [int(atom.GetProp('Acid'))] + \
                      [int(atom.GetProp('Halogen'))]
        if scaffold:
            results = results + [int(atom.GetProp('Scaffold'))]
        feat.append(results)

    return np.array(feat)


def bond_attr(mol, use_chirality=True):
    feat = []
    index = []
    n = mol.GetNumAtoms()
    for i in range(n):
        for j in range(n):
            if i != j:
                bond = mol.GetBondBetweenAtoms(i, j)
                if bond is not None:
                    bt = bond.GetBondType()
                    bond_feats = [
                        bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE,
                        bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC,
                        bond.GetIsConjugated(),
                        bond.IsInRing()
                    ]
                    if use_chirality:
                        bond_feats = bond_feats + onehot_encoding_unk(
                            str(bond.GetStereo()),
                            ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"])
                    feat.append(bond_feats)
                    index.append([i, j])

    return np.array(index), np.array(feat)


def bond_break(mol):
    results = np.array(sorted(list(FindBRICSBonds(mol))), dtype=np.int64)
    print(results)
    if results.size == 0:
        cluster_idx = []
        Chem.rdmolops.GetMolFrags(mol, asMols=True, frags=cluster_idx)
        fra_edge_index, fra_edge_attr = bond_attr(mol)

    else:
        bond_to_break = results[:, 0, :]
        bond_to_break = bond_to_break.tolist()
        with Chem.RWMol(mol) as rwmol:
            for i in bond_to_break:
                rwmol.RemoveBond(*i)
        rwmol = rwmol.GetMol()
        cluster_idx = []
        Chem.rdmolops.GetMolFrags(rwmol, asMols=True, sanitizeFrags=False, frags=cluster_idx)
        fra_edge_index, fra_edge_attr = bond_attr(rwmol)
        cluster_idx = torch.LongTensor(cluster_idx)

    return fra_edge_index, fra_edge_attr, cluster_idx
fun_smarts = {
        'Hbond_donor': '[$([N;!H0;v3,v4&+1]),$([O,S;H1;+0]),n&H1&+0]',
        'Hbond_acceptor': '[$([O,S;H1;v2;!$(*-*=[O,N,P,S])]),$([O,S;H0;v2]),$([O,S;-]),$([N;v3;!$(N-*=[O,N,P,S])]),n&X2&H0&+0,$([o,s;+0;!$([o,s]:n);!$([o,s]:c:n)])]',
        'Basic': '[#7;+,$([N;H2&+0][$([C,a]);!$([C,a](=O))]),$([N;H1&+0]([$([C,a]);!$([C,a](=O))])[$([C,a]);!$([C,a](=O))]),$([N;H0&+0]([C;!$(C(=O))])([C;!$(C(=O))])[C;!$(C(=O))]),$([n;X2;+0;-0])]',
        'Acid': '[C,S](=[O,S,P])-[O;H1,-1]',
        'Halogen': '[F,Cl,Br,I]'
        }
FunQuery = dict([(pharmaco, Chem.MolFromSmarts(s)) for (pharmaco, s) in fun_smarts.items()])


def tag_pharmacophore(mol):
    for fungrp, qmol in FunQuery.items():
        matches = mol.GetSubstructMatches(qmol)
        match_idxes = []
        for mat in matches:
            match_idxes.extend(mat)
        for i, atom in enumerate(mol.GetAtoms()):
            tag = '1' if i in match_idxes else '0'
            atom.SetProp(fungrp, tag)
    return mol


# tag scaffold information to each atom
def tag_scaffold(mol):
    core = MurckoScaffold.GetScaffoldForMol(mol)
    match_idxes = mol.GetSubstructMatch(core)
    for i, atom in enumerate(mol.GetAtoms()):
        tag = '1' if i in match_idxes else '0'
        atom.SetProp('Scaffold', tag)
    return mol
class MolData(Data):
    def __init__(self, fra_edge_index=None, fra_edge_attr=None, cluster_index=None, **kwargs):
        super(MolData, self).__init__(**kwargs)
        self.cluster_index = cluster_index
        self.fra_edge_index = fra_edge_index
        self.fra_edge_attr = fra_edge_attr

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'cluster_index':
            return int(self.cluster_index.max()) + 1
        else:
            return super().__inc__(key, value, *args, **kwargs)


class MolDataset(InMemoryDataset):

    def __init__(self, root, dataset, task_type, tasks, logger=None,
                 transform=None, pre_transform=None, pre_filter=None):

        self.tasks = tasks
        self.dataset = dataset
        self.task_type = task_type

        super(MolDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['{}.csv'.format(self.dataset)]

    @property
    def processed_file_names(self):
        return ['{}.pt'.format(self.dataset)]

    def download(self):
        pass

    def process(self):
        df = pd.read_csv(self.raw_paths[0])
        smilesList = df.smiles.values
        print(f'number of all smiles: {len(smilesList)}')
        remained_smiles = []
        canonical_smiles_list = []
        for smiles in smilesList:
            try:
                canonical_smiles_list.append(Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=True))
                remained_smiles.append(smiles)
            except:
                print(f'not successfully processed smiles: {smiles}')
                pass
        print(f'number of successfully processed smiles: {len(remained_smiles)}')

        df = df[df["smiles"].isin(remained_smiles)].reset_index()
        target = df[self.tasks].values
        smilesList = df.smiles.values
        data_list = []

        for i, smi in enumerate(tqdm(smilesList)):

            mol = Chem.MolFromSmiles(smi)
            data = self.mol2graph(mol)

            if data is not None:
                label = target[i]

                data.y = torch.LongTensor([label])
                if self.task_type == 'regression':
                    data.y = torch.FloatTensor([label])

                data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def mol2graph(self, mol):
        smiles = Chem.MolToSmiles(mol)
        if mol is None: return None
        node_attr = atom_attr(mol)
        edge_index, edge_attr = bond_attr(mol)
        fra_edge_index, fra_edge_attr, cluster_index = bond_break(mol)
        data = MolData(
            x=torch.FloatTensor(node_attr),
            edge_index=torch.LongTensor(edge_index).t(),
            edge_attr=torch.FloatTensor(edge_attr),
            fra_edge_index=torch.LongTensor(fra_edge_index).t(),
            fra_edge_attr=torch.FloatTensor(fra_edge_attr),
            cluster_index=torch.LongTensor(cluster_index),
            y=None,
            smiles=smiles,
        )
        return data






device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
def train(model, optimizer,train_loader):
    train_labels = 0
    train_predictions = 0
    total_loss = total_examples = 0
    criterion = torch.nn.BCELoss(
    weight=None,
    size_average=None,
    reduction="mean",
)
    train_label=[]
    train_pre=[]
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out ,att= model(data)
        # out = model(data)

        y = data.y.view([-1])


        out1=out
        out1 = out1.view([-1])


        for i in out1:
            i = float(i)
            train_pre.append(i)
        for j in y:
            j = float(j)
            train_label.append(j)


        # print("train : ", y.shape)


        out1 = out1.float()

        loss = criterion(out1.float(),y.float())

        loss.backward()
        optimizer.step()
#         train_labels += train_labels + y
#         train_predictions += train_predictions + out1
        total_loss += float(loss) * data.num_graphs
        total_examples += data.num_graphs
      #  pre = np.around(train_pre, 0).astype(int)
       # acc_train = sklearn.metrics.accuracy_score(pre, train_labels, normalize=True, sample_weight=None)
    return total_loss,train_pre,train_label

@torch.no_grad()
def test(loader, model):
    # mse = []
    model.eval()
    label1=[]
    pre1=[]
    total_loss = total_examples = 0
    criterion = torch.nn.BCEWithLogitsLoss(
        weight=None,
        size_average=None,
        reduction="mean",
    )
    for data in loader:
        data = data.to(device)
        out ,att= model(data)

        # out = model(data)
        # mse.append(F.mse_loss(out, data.y, reduction='none').cpu())
        # return float(torch.cat(mse, dim=0).mean().sqrt())

        y = data.y.view([-1])
        out1=out
        out1 = out1.view([-1])
        y = torch.tensor(y, dtype=float)
        out1 = torch.tensor(out1, dtype=float)

        for i in out1:
            i = float(i)
            pre1.append(i)
        for j in y:
            j = float(j)
            label1.append(j)

        # print("test : ", y.shape)
        test_loss =criterion(y, out1)
        # print("no of graphs: ", data.num_graphs)
        total_loss += float(test_loss) * data.num_graphs
        total_examples += data.num_graphs
#         total_preds = torch.cat((total_preds, out1.cpu()), 0)
#         total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
        # mse.append(test_loss).cpu()
    # return test_loss,float(torch.cat(mse, dim=0).mean().sqrt())
        auc=roc_auc_score(label1,pre1)
        pre = np.around(pre1, 0).astype(int)
        acc=sklearn.metrics.accuracy_score(pre, label1, normalize=True, sample_weight=None)

    return total_loss,acc,pre1,label1 #,total_labels.numpy().flatten(),total_preds.numpy().flatten()
epochs=500
seed = 145
epochs=500
seed = 145
torch.manual_seed(seed)

the_last_loss = 100
patience =6000
trigger_times = 0
count_loss_difference = 0
learning_rate = 0.005
weight_decay=0.000307616688331247

train_losses = []
test_losses = []
test_accs = []
val_losses=[]
best_ret=[]
kf = KFold(n_splits=5, shuffle=True, random_state=0)
# train_fold = []

# for epoch in range(20):
import csv
def get_header(path):
    with open(path) as f:
        header = next(csv.reader(f))

    return header


def get_task_names(path, use_compound_names=False):
    index = 2 if use_compound_names else 1
    task_names = get_header(path)[index:]

    return task_names

import csv
def get_header(path):
    with open(path) as f:
        header = next(csv.reader(f))

    return header


def get_task_names(path, use_compound_names=False):
    index = 2 if use_compound_names else 1
    task_names = get_header(path)[index:]

    return task_names




train_task_names = get_task_names('data/precessed/raw/bbbp_train.csv')
dataset = 'bbbp_train'
task_type = 'classfication'
tasks = train_task_names
train_data = MolDataset( root='data/precessed',dataset=dataset, task_type=task_type, tasks=tasks)

test_task_names=get_task_names('data/precessed/raw/bbbp_test.csv')
test_dataset = 'bbbp_test'
task_type = 'classification'
test_tasks = test_task_names
test_data = MolDataset( root='data/precessed',dataset=test_dataset, task_type=task_type, tasks=test_tasks)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=True)
i=0
model1 = HiGNN(in_channels=46,
               hidden_channels=128,
               out_channels=1,
               edge_dim=10,
               num_layers=3,
               dropout=0.4,
               slices=1,

               brics=True
               )
model1.to(device)
#####################################################
'''

for epoch in range(epochs):


    optimizer = torch.optim.Adam(model1.parameters(), lr=learning_rate,
                                 weight_decay=weight_decay)
    train_loss, train_acc, label2 = train(model1, optimizer, train_loader)
    test_loss, test_acc, pre, label = test(test_loader, model1)
    #         score = metrics.r2_score(true, prediction)
    #         , true, prediction

    print(f'Epoch: {epoch:03d}, Loss: {train_loss:.4f} '
          f'Test: {test_acc:.4f} ')  # f'score: {score:.4f} '


    if epoch ==60 :

    #    train_series = pd.Series(train_losses)
     ##   val_series = pd.Series(val_losses)
       # frame = {'train': train_series, 'test': val_series}
        #output1 = pd.DataFrame(frame)

        # print(output)
        #  output.to_csv(str(i)+"bbbp.csv")
        torch.save(model1, 'bbbp_tsne80.pt')
    # torch.save(model.state_dict(), model_file_name)
'''


test_task_names='bbbp_test'
test_dataset = 'bbbp_test'
task_type = 'classification'
test_tasks = test_task_names
test_data = MolDataset( root='data/precessed',dataset=test_dataset, task_type=task_type, tasks=test_tasks)



dataloader= DataLoader(test_data,batch_size=1,shuffle=True)
i=0
model = torch.load('bbbp_tsne80.pt')
model.to(device)
model1 = HiGNN(in_channels=46,
               hidden_channels=128,
               out_channels=1,
               edge_dim=10,
               num_layers=3,
               dropout=0.4,
               slices=1,

               brics=True
               )
model1.to(device)
model1.to(device)
embs=[]
labels=[]
for data in dataloader:
    data=data.to(device)
    pre,emb=model(data)

    label=data.y.cpu()

    emb=emb.cpu()
    embs.append(emb.detach().numpy())
    labels.append(label.detach  ().numpy())
#
embs = np.concatenate(embs)
print(embs.shape)
labels = np.concatenate(labels)
print(labels.shape)
print('............................')


import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.manifold import TSNE
tsne = TSNE(n_components=3,  learning_rate=200, init='pca', random_state=42,perplexity=25)
X_3d=tsne.fit_transform(embs)
#绘制3d图形
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 定义颜色列表（这里只使用了两个颜色）
css4 = list(mcolors.CSS4_COLORS.keys())
color_ind = [60,65]
css4 = [css4[v] for v in color_ind]
css4=['blue','red']
for lbi in range(2):
    temp = X_3d[labels==lbi]

    ax.scatter(temp[:, 0], temp[:, 1], temp[:, 2], '.', color=css4[int(lbi)], label=f'Class {int(lbi)}')
plt.legend()

# 显示图表
plt.show()

plt.savefig('output/bbbpPretrained2.png')
plt.show()
'''












#cluster_index=[296], fra_edge_index=[2, 580], fra_edge_attr=[580, 10], y=[16], batch=[296], ptr=[17])

