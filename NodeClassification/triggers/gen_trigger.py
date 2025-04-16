import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.nn as geo_nn

from triggers.trojan_net import GraphTrojanNet
from triggers.position import obtain_attach_nodes, cluster_distance_selection, \
    cluster_degree_selection, obtain_attach_nodes_degree, obtain_attach_nodes_cluster


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

    def __init__(self, feat_dim, hid_dim, trigger_size, tau=0.01):
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

    def forward(self, feat, edge_index, target_idx):

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

        return n_feat, n_edge_index, n_edge_attrs


class GenTrigger(object):

    def __init__(self, trigger_size, trigger_position, poisoning_intensity, feat_dim, dataset_name, num_classes, target_class,
                 num_hidden_generator, dis_weight, num_hidden, dropout, train_epochs, lr, weight_decay, tau, seed, device):

        # self.trigger_model = GenModel(
        #     feat_dim, num_hidden_generator, trigger_size, tau
        # ).to(device)
        self.trigger_model = GraphTrojanNet(
            feat_dim, num_hidden_generator, trigger_size, tau
        ).to(device)
        self.optimizer = torch.optim.Adam(self.trigger_model.parameters(), lr=lr, weight_decay=weight_decay)
        self.trigger_position = trigger_position

        self.num_poisoned_nodes = 0
        self.poisoning_density = poisoning_intensity

        self.seed = seed
        self.dataset_name, self.num_hidden, self.num_classes,  self.dropout, self.lr, self.weight_decay, self.train_epochs, self.dis_weight, self.target_class, self.device = \
            dataset_name, num_hidden, num_classes,  dropout, lr, weight_decay, train_epochs, dis_weight, target_class, device

    def gen_idx_attach(self, data, train_edge_index, idx_train, idx_val, idx_clean_test, idx_unlabelled):
        # Pick-up nodes to attach triggers
        self.num_poisoned_nodes = int(len(idx_unlabelled) * self.poisoning_density)
        if self.trigger_position == 'random':
            idx_attach = obtain_attach_nodes(idx_unlabelled, self.num_poisoned_nodes)
        elif self.trigger_position == 'learn_cluster':
            idx_attach = cluster_distance_selection(
                data, idx_train, idx_val, idx_clean_test, idx_unlabelled,
                train_edge_index, self.num_poisoned_nodes, self.dataset_name, self.num_hidden,
                self.num_classes, self.dropout, self.lr, self.weight_decay, self.train_epochs, self.dis_weight, self.target_class, self.device
            )
        elif self.trigger_position == 'learn_cluster_degree':
            idx_attach = cluster_degree_selection(
                data, idx_train, idx_val, idx_clean_test, idx_unlabelled,
                train_edge_index, self.num_poisoned_nodes, self.dataset_name, self.num_hidden,
                self.num_classes, self.dropout, self.lr, self.weight_decay, self.train_epochs, self.dis_weight, self.target_class, self.seed, self.device
            )
        elif self.trigger_position == 'degree':
            idx_attach = obtain_attach_nodes_degree(idx_unlabelled, train_edge_index, data.x.shape[0], self.num_poisoned_nodes)
        elif self.trigger_position == 'cluster':
            idx_attach = obtain_attach_nodes_cluster(idx_unlabelled, train_edge_index, data.x.shape[0], self.num_poisoned_nodes)
        else:
            raise NotImplementedError('Trigger Position {} is Not Defined!'.format(self.trigger_position))
        idx_attach = torch.LongTensor(idx_attach).to(self.device)

        return idx_attach

    @torch.no_grad()
    def get_poisoned(self, feat, edge_index, edge_weight, labels, idx_attach):
        self.trigger_model.eval()

        injected_feat, injected_edge_index, injected_edge_weight = self.trigger_model(feat, edge_index, idx_attach)

        injected_feat = torch.cat([feat, injected_feat], dim=0)
        injected_edge_weight = torch.cat([edge_weight, injected_edge_weight], dim=0)
        injected_edge_index = torch.cat([edge_index, injected_edge_index], dim=1)

        labels[idx_attach] = self.target_class
        return injected_feat, injected_edge_index, injected_edge_weight, labels

    def poison_train_graph(self, train_data, idx_attach):
        train_data.x, train_data.edge_index, train_data.edge_weight, train_data.y = self.get_poisoned(
            train_data.x, train_data.edge_index, train_data.edge_weight, train_data.y, idx_attach
        )

        train_mask = torch.zeros_like(train_data.train_mask, dtype=torch.bool, device=self.device)
        train_mask[idx_attach] = True
        train_data.train_mask = torch.bitwise_or(train_data.train_mask, train_mask)

        return train_data

    def poison_test_graph(self, test_data, idx_attack):
        test_data.x, test_data.edge_index, test_data.edge_weight, test_data.y = self.get_poisoned(
            test_data.x, test_data.edge_index, test_data.edge_weight, test_data.y, idx_attack
        )

        test_mask = torch.zeros_like(test_data.test_mask, dtype=torch.bool, device=self.device)
        test_mask[idx_attack] = True
        test_data.test_mask = torch.bitwise_or(test_data.test_mask, test_mask)
        return test_data
