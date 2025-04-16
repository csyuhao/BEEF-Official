import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn.conv import GCNConv


class GCN_Encoder(nn.Module):

    def __init__(self, num_feat, num_hidden, num_classes,
                 dropout=0.5, layer=2, use_ln=False, layer_norm_first=False):
        super(GCN_Encoder, self).__init__()

        self.use_ln = use_ln
        self.dropout = dropout
        self.num_feat = num_feat
        self.num_classes = num_classes
        self.layer_norm_first = layer_norm_first

        self.body = GCN_body(num_feat, num_hidden, dropout, layer, use_ln=use_ln, layer_norm_first=layer_norm_first)
        self.fc = nn.Linear(num_hidden, num_classes)

    def forward(self, x, edge_index, edge_weight=None):
        x = self.body(x, edge_index, edge_weight)
        return self.fc(x)

    def get_h(self, x, edge_index, edge_weight):
        x = self.body(x, edge_index, edge_weight)
        return x


class GCN_body(nn.Module):

    def __init__(self, num_feat, num_hidden, dropout=0.5, num_layers=2, layer_norm_first=False, use_ln=False):
        super(GCN_body, self).__init__()
        self.num_feat = num_feat
        self.dropout = dropout

        self.layers, self.layer_norms = nn.ModuleList(), nn.ModuleList()
        for idx in range(num_layers):
            if idx == 0:
                self.layer_norms.append(nn.LayerNorm(num_feat))
                self.layers.append(GCNConv(num_feat, num_hidden))
                continue
            self.layer_norms.append(nn.LayerNorm(num_hidden))
            self.layers.append(GCNConv(num_hidden, num_hidden))

        self.layer_norm_first = layer_norm_first
        self.use_ln = use_ln

    def forward(self, x, edge_index, edge_weight=None):
        edge_index, edge_weight = edge_index, edge_weight

        if self.layer_norm_first:
            x = self.layer_norms[0](x)

        i = 0
        for idx, layer in enumerate(self.layers):
            x = layer(x, edge_index, edge_weight)
            if idx == len(self.layers) - 1:
                break
            x = F.relu(x)
            if self.use_ln:
                x = self.layer_norms[i + 1](x)
            x = F.dropout(x, self.dropout, training=self.training)
            i += 1
        return x
