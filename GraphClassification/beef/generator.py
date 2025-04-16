

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.nn as geo_nn
from torch_geometric.data import Data
from torch_geometric.utils import degree


class GumbelSoftmax(object):

    def __init__(self):
        pass

    def sample(self, logits, mask=None, tau=0.01):

        gumbels = (
            -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
        )
        gumbels = (logits + gumbels) / tau

        if mask is not None:
            minimal = torch.min(gumbels, dim=-1, keepdim=True)[0]
            gumbels = gumbels * mask + (1 - mask) * minimal

        sample_one_hot = gumbels.softmax(dim=-1)
        return sample_one_hot


class ReshapeLayer(nn.Module):

    def __init__(self, target_shape, reduction='none', keepdim=False):
        super(ReshapeLayer, self).__init__()
        self.target_shape = target_shape
        self.reduction = reduction
        self.keepdim = keepdim

    def forward(self, x):
        x = x.reshape(self.target_shape)
        if self.reduction == 'mean':
            x = torch.mean(x, dim=0, keepdim=self.keepdim)
        return x


class NodeSimilarityLayer(nn.Module):

    def __init__(self, normalized=True):
        super(NodeSimilarityLayer, self).__init__()
        self.normalized = normalized

    def forward(self, x):
        if self.normalized is True:
            x = F.normalize(x, p=2, dim=-1)

        num_target_nodes, num_trigger_nodes = x.shape[:2]
        x = (torch.bmm(x, x.transpose(1, 2)) + 1.) * 0.5
        indices = torch.triu_indices(num_trigger_nodes, num_trigger_nodes, offset=1)

        prob_matrix = torch.zeros(
            (num_target_nodes, int(num_trigger_nodes * (num_trigger_nodes - 1)) // 2, 2), device=x.device, dtype=torch.float32
        )
        prob_matrix[:, :, 1] = x[:, indices[0], indices[1]]
        prob_matrix[:, :, 0] = 1 - prob_matrix[:, :, 1]
        return prob_matrix


class GenModel(nn.Module):

    def __init__(self, feat_dim, hid_dim, trigger_size, tau=0.1):
        super(GenModel, self).__init__()

        self.feat_dim = feat_dim
        self.hid_dim = hid_dim
        self.trigger_size = trigger_size
        self.tau = tau

        self.sampler = GumbelSoftmax()

        adj_matrix = torch.triu(torch.ones(size=(trigger_size, trigger_size), dtype=torch.long), diagonal=1)
        self.trigger_edge_index = torch.nonzero(adj_matrix, as_tuple=False).t().contiguous()
        self.trigger_edge_attrs = torch.ones(size=(self.trigger_edge_index.shape[1],), dtype=torch.float)

        self.encoder_layers = nn.Sequential(*[
            geo_nn.GCNConv(in_channels=feat_dim, out_channels=int(self.trigger_size * hid_dim)),
            nn.LayerNorm(int(self.trigger_size * hid_dim)),
            nn.ReLU(),
            geo_nn.GCNConv(in_channels=int(self.trigger_size * hid_dim), out_channels=int(self.trigger_size * hid_dim)),
            nn.LayerNorm(int(self.trigger_size * hid_dim)),
            nn.ReLU(),
            ReshapeLayer(target_shape=(-1, self.trigger_size, hid_dim))
        ])

        self.attr_output_layer = nn.Sequential(*[
            ReshapeLayer(target_shape=(-1, hid_dim)),
            nn.Linear(in_features=hid_dim, out_features=feat_dim),
        ])

        self.edge_output_layer = nn.Sequential(*[
            NodeSimilarityLayer(normalized=True)
        ])

    def sample_edges(self, sample_weight, feat, target_idx):
        # Change device
        self.trigger_edge_index = self.trigger_edge_index.to(feat.device)

        # Number of target nodes
        num_target_nodes = len(target_idx)

        # Sample new edges according to probabilities
        injected_edge_weights = self.sampler.sample(sample_weight, tau=self.tau)
        injected_edge_weights = injected_edge_weights[:, :, 1].reshape(-1)

        # Expand edges of within triggers
        trigger_edge_index = torch.cat([self.trigger_edge_index] * num_target_nodes, dim=1)

        base_injected_node_idx = torch.arange(
            start=0, end=num_target_nodes * self.trigger_size, step=self.trigger_size, dtype=torch.float32, device=feat.device
        ) + feat.shape[0]
        expanded_injected_nodes = base_injected_node_idx.reshape(-1, 1).repeat(1, self.trigger_edge_index.shape[1]).reshape(-1)

        row, col = trigger_edge_index
        row = row + expanded_injected_nodes
        col = col + expanded_injected_nodes
        trigger_edge_attrs = injected_edge_weights.reshape(-1)
        # trigger_edge_attrs = torch.cat([self.trigger_edge_attrs.to(feat.device)] * num_target_nodes, dim=0)

        connected_node_idx = torch.arange(
            start=0, end=num_target_nodes * self.trigger_size, step=1, dtype=torch.float32, device=feat.device
        ) + feat.shape[0]
        expanded_target_idx = target_idx.reshape(-1, 1).repeat(1, self.trigger_size).reshape(-1)
        connected_edge_index = torch.concat(
            [
                torch.concat([connected_node_idx.reshape(1, -1), expanded_target_idx.reshape(1, -1)], dim=1),
                torch.concat([expanded_target_idx.reshape(1, -1), connected_node_idx.reshape(1, -1)], dim=1)
            ],
            dim=0
        )
        connected_edge_attrs = torch.ones(size=(connected_edge_index.shape[1],), device=feat.device, dtype=torch.float)

        n_edge_index = torch.concat(
            [connected_edge_index, torch.stack([row, col], dim=0), torch.stack([col, row], dim=0)],
            dim=1
        )
        n_edge_attrs = torch.concat(
            [connected_edge_attrs, trigger_edge_attrs, trigger_edge_attrs],
            dim=0
        )

        effective_idx = n_edge_attrs >= 0.5
        row, col = n_edge_index
        n_edge_index = torch.stack([row[effective_idx], col[effective_idx]], dim=0)
        n_edge_attrs = n_edge_attrs[effective_idx]
        return n_edge_index.long(), n_edge_attrs

    def forward(self, feat, edge_index):
        feat = F.normalize(feat, p=2, dim=-1)
        node_degree = degree(edge_index[0], num_nodes=feat.shape[0])
        min_node_degree = torch.min(node_degree)
        target_idx = torch.nonzero(node_degree == min_node_degree, as_tuple=False).reshape(-1)

        attr_embedding = feat
        for layer in self.encoder_layers:
            if isinstance(layer, geo_nn.GCNConv):
                attr_embedding = layer(attr_embedding, edge_index)
            else:
                attr_embedding = layer(attr_embedding)
        attr_embedding = attr_embedding[target_idx]

        n_feat = self.attr_output_layer(attr_embedding)

        edge_weights = self.edge_output_layer(attr_embedding)
        n_edge_index, n_edge_attrs = self.sample_edges(edge_weights, feat, target_idx)

        n_feat = torch.cat([feat, n_feat], dim=0)
        n_edge_index = torch.cat([edge_index, n_edge_index], dim=1)
        n_edge_attrs = torch.cat([torch.ones(size=(edge_index.shape[1],), device=feat.device, dtype=torch.float), n_edge_attrs], dim=0)

        data = Data(x=n_feat, edge_index=n_edge_index, edge_weight=n_edge_attrs, device=n_feat.device)
        return data
