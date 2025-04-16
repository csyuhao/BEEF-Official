import torch.nn as nn
import torch_geometric.nn as gnn


class APPNP(nn.Module):
    def __init__(self,
                 n_feat, n_hid, n_class, num_layers, dropout):

        super().__init__()

        layers = []
        for idx in range(num_layers):
            if idx == 0:
                layers.append(nn.Linear(n_feat, n_hid, bias=True))
            elif idx == num_layers - 1:
                layers.append(nn.Linear(n_hid, n_class, bias=True))
            else:
                layers.append(nn.Linear(n_hid, n_hid, bias=True))

            if idx != num_layers - 1:
                layers.append(nn.LayerNorm(n_hid))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))

        layers.append(gnn.APPNP(K=3, alpha=0.5))
        self.layers = nn.Sequential(*layers)

    def forward(self, x, edge_index, edge_weight=None, is_proxy=False):
        final_proxy = None
        for layer in self.layers:
            if isinstance(layer, gnn.APPNP):
                x = layer(x, edge_index, edge_weight)
            elif isinstance(layer, nn.Linear):
                x = layer(x)
                final_proxy = x
            else:
                x = layer(x)
        if is_proxy:
            return final_proxy
        return x
