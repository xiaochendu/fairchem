import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool

from cgcnn.layers import CGCNNConv


class CGCNN(nn.Module):
    def __init__(
        self,
        num_atoms,
        bond_feat_dim,
        num_targets,
        atom_embedding_size=64,
        num_graph_conv_layers=6,
        fc_feat_size=128,
        num_fc_layers=4,
    ):
        super(CGCNN, self).__init__()
        self.embedding = nn.Linear(num_atoms, atom_embedding_size)

        self.convs = nn.ModuleList(
            [
                CGCNNConv(node_dim=atom_embedding_size, edge_dim=bond_feat_dim)
                for _ in range(num_graph_conv_layers)
            ]
        )

        self.conv_to_fc = nn.Sequential(
            nn.Linear(atom_embedding_size, fc_feat_size), nn.Softplus()
        )

        if num_fc_layers > 1:
            layers = []
            for _ in range(num_fc_layers - 1):
                layers.append(nn.Linear(fc_feat_size, fc_feat_size))
                layers.append(nn.Softplus())
            self.fcs = nn.Sequential(*layers)
        self.fc_out = nn.Linear(fc_feat_size, num_targets)

    def forward(self, data):
        node_feats = self.embedding(data.x)
        for f in self.convs:
            node_feats = f(node_feats, data.edge_index, data.edge_attr)
        mol_feats = global_mean_pool(node_feats, data.batch)
        mol_feats = self.conv_to_fc(mol_feats)
        if hasattr(self, "fcs"):
            mol_feats = self.fcs(mol_feats)
        out = self.fc_out(mol_feats)
        return out