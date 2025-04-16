import torch.nn as nn

from torch_geometric.nn.conv import GCNConv

from models.layers import MaskedGCNConv, MaskedLinear


class GCN(nn.Module):

    def __init__(self, num_feat, num_hidden, num_classes, dropout=0.5, layer=2, masked=False, l1=1e-3, mask_one_init=False):
        super(GCN, self).__init__()

        self.layers = nn.ModuleList()
        for idx in range(layer):
            if idx == 0:
                self.layers.append(
                    GCNConv(num_feat, num_hidden)
                    if not masked else MaskedGCNConv(num_feat, num_hidden, l1=l1, mask_one_init=mask_one_init)
                )
            else:
                self.layers.append(
                    GCNConv(num_hidden, num_hidden)
                    if not masked else MaskedGCNConv(num_hidden, num_hidden, l1=l1, mask_one_init=mask_one_init)
                )
            self.layers.extend([
                nn.LayerNorm(num_hidden),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
        self.layers.append(
            nn.Linear(num_hidden, num_classes)
            if not masked else MaskedLinear(num_hidden, num_classes, l1=l1, mask_one_init=mask_one_init)
        )

    def forward(self, x, edge_index, edge_weight=None, is_proxy=False):
        final_proxy = None
        for layer in self.layers:
            if isinstance(layer, GCNConv) or isinstance(layer, MaskedGCNConv):
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
