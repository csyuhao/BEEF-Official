import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as geo_nn


class GNNGuard(nn.Module):

    def __init__(self, n_feat, n_hid, n_class, dropout=0.5, use_ln=True):
        super(GNNGuard, self).__init__()

        self.n_feat = n_feat
        self.hidden_sizes = [n_hid]
        self.n_class = n_class
        self.dropout = dropout

        self.use_ln = use_ln
        if use_ln:
            self.lns = nn.ModuleList()
            self.lns.append(torch.nn.LayerNorm(n_feat))
            self.lns.append(nn.LayerNorm(n_hid))

        """GCN from geometric"""
        """network from torch-geometric, """
        self.gc1 = geo_nn.GCNConv(n_feat, n_hid, bias=True)
        self.gc2 = geo_nn.GCNConv(n_hid, n_hid, bias=True)
        self.fc = nn.Linear(n_hid, n_class)

    def forward(self, x, adj, edge_weight=None, is_proxy=False):
        """we don't change the edge_index, just update the edge_weight;
        some edge_weight are regarded as removed if it equals to zero"""

        """GCN and GTA"""

        if self.use_ln:
            x = self.lns[0](x)
        edge_weight = self.att_coef(x, adj)
        x = self.gc1(x, adj, edge_weight=edge_weight)
        x = F.relu(x)
        if self.use_ln:
            x = self.lns[1](x)

        edge_weight = self.att_coef(x, adj)

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj, edge_weight=edge_weight)

        if is_proxy:
            return x

        x = self.fc(F.relu(x))

        return x

    def initialize(self):
        self.gc1.reset_parameters()
        self.gc2.reset_parameters()

    def att_coef(self, fea, edge_index):
        fea = fea.detach()
        sim = torch.cosine_similarity(fea[edge_index[0]], fea[edge_index[1]])
        sim[sim < 0.1] = 0.0
        return sim
