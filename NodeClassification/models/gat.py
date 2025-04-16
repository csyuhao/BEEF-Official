import torch.nn as nn

from torch_geometric.nn.conv import GATConv

from models.layers import MaskedGATConv, MaskedLinear


class GAT(nn.Module):

    def __init__(self, num_feat, num_hidden, num_classes, n_heads=8, dropout=0.5, masked=False, l1=1e-3, mask_one_init=False):
        super(GAT, self).__init__()
        self.layers = nn.Sequential(*[
            GATConv(num_feat, num_hidden, heads=n_heads, dropout=dropout)
            if masked is False else MaskedGATConv(num_feat, num_hidden, heads=n_heads, dropout=dropout, l1=l1, mask_one_init=mask_one_init),
            nn.LayerNorm(num_hidden * n_heads),
            nn.ReLU(),
            GATConv(num_hidden * n_heads, num_hidden, heads=1, concat=False, dropout=dropout)
            if masked is False else MaskedGATConv(num_hidden * n_heads, num_hidden,
                                                  heads=1, concat=False,dropout=dropout, l1=l1, mask_one_init=mask_one_init),
            nn.LayerNorm(num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_classes) if masked is False else MaskedLinear(num_hidden, num_classes, l1=l1, mask_one_init=mask_one_init)
        ])

    def forward(self, x, edge_index, edge_weight=None, is_proxy=False):

        final_proxy = None
        for layer in self.layers:
            if isinstance(layer, GATConv) or isinstance(layer, MaskedGATConv):
                x = layer(x, edge_index, edge_weight)
                final_proxy = x
            else:
                x = layer(x)
        if is_proxy:
            return final_proxy
        return x

    def reset_parameters(self):
        """ Initialize parameters of GCN.
        """
        for layer in self.layers:
            layer.reset_parameters()
