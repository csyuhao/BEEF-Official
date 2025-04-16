import networkx as nx
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from models.gcn import GCN


class GraphTrojanNet(nn.Module):

    def __init__(self, num_feat, num_out, num_hidden=32, num_layers=1, dropout=0.00):
        super(GraphTrojanNet, self).__init__()
        layers = []

        # Why to delete negative features?
        # if dropout > 0:
        #     layers.append(nn.Dropout(p=dropout))

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(num_feat, num_feat))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))

        self.layers = nn.Sequential(*layers)

        if num_feat > 3000:
            self.feat = nn.Sequential(*[
                nn.Linear(num_feat, num_hidden),
                nn.ReLU(),
                nn.Linear(num_hidden, num_out * num_feat)
            ])
        else:
            self.feat = nn.Linear(num_feat, num_out * num_feat)

    def forward(self, inputs):
        """
        'input', 'mask' and 'thrd', should already in cuda before sent to this function.
        If using sparse format, corresponding tensor should already in sparse format before
        sent into this function
        """
        h = self.layers(inputs)
        feat = self.feat(h)
        return feat


class HomoLoss(nn.Module):
    def __init__(self):
        super(HomoLoss, self).__init__()
        pass

    @staticmethod
    def forward(trigger_edge_index, trigger_edge_weights, x, thrd):
        trigger_edge_index = trigger_edge_index[:, trigger_edge_weights > 0.0]
        edge_sims = F.cosine_similarity(x[trigger_edge_index[0]], x[trigger_edge_index[1]])

        loss = torch.relu(thrd - edge_sims).mean()
        return loss


class HeuristicBackdoor:

    def __init__(self, target_class, num_hidden, thrd, trigger_size, trigger_type, density, degree,
                 lr, weight_decay, outer_epochs, inner_epochs, num_classes, device):
        self.device = device
        self.thrd = thrd
        self.num_hidden = num_hidden
        self.target_class = target_class
        self.trigger_size = trigger_size
        self.lr, self.weight_decay = lr, weight_decay
        self.inner_epochs, self.outer_epochs = inner_epochs, outer_epochs
        self.trojan_model, self.shadow_model, self.num_classes = None, None, num_classes
        self.trigger_index = self.get_trigger_index(trigger_size, trigger_type, density, degree)

    def init_models(self, features):
        # initialize a shadow model
        self.shadow_model = GCN(
            num_feat=features.shape[1], num_hidden=self.num_hidden, num_classes=self.num_classes, dropout=0.0
        ).to(self.device)

        # initialize a trojanNet to generate trigger
        self.trojan_model = GraphTrojanNet(num_feat=features.shape[1], num_out=self.trigger_size, num_hidden=self.num_hidden, num_layers=2).to(self.device)

    @staticmethod
    def get_trigger_index(trigger_size, trigger_type, density, degree):
        print('Start generating trigger by {}'.format(trigger_type))
        graph_trigger = None
        if trigger_type == 'renyi':
            graph_trigger = nx.erdos_renyi_graph(trigger_size, density, directed=False)
            if graph_trigger.edges():
                graph_trigger = graph_trigger
            else:
                graph_trigger = nx.erdos_renyi_graph(trigger_size, 1.0, directed=False)
        elif trigger_type == 'ws':
            graph_trigger = nx.watts_strogatz_graph(trigger_size, degree, density)
        elif trigger_type == 'ba':

            if degree >= trigger_size:
                degree = trigger_size - 1

            # n: int Number of nodes
            # m: int Number of edges to attach from a new node to existing nodes
            graph_trigger = nx.random_graphs.barabasi_albert_graph(n=trigger_size, m=degree)
        elif trigger_type == 'rr':
            # d int The degree of each node.
            # n integer The number of nodes.The value of must be even.

            if degree >= trigger_size:
                degree = trigger_size - 1
            if trigger_size % 2 != 0:
                trigger_size += 1

            # generate a regular graph which has 20 nodes & each node has 3 neighbour nodes.
            graph_trigger = nx.random_graphs.random_regular_graph(d=degree, n=trigger_size)

        # Convert the graph to an edge list in COO format
        edge_list = np.array(list(graph_trigger.edges())).T

        # Insert [0, 0] at the beginning of the edge list
        edge_list = np.insert(edge_list, 0, [0, 0], axis=1)
        edge_index = torch.tensor(edge_list, dtype=torch.long)
        return edge_index

    def get_trojan_edge(self, start, idx_attach, trigger_size):
        edge_list = []

        for idx in idx_attach:
            edges = self.trigger_index.clone()
            edges[0, 0] = idx
            edges[1, 0] = start
            edges[:, 1:] = edges[:, 1:] + start

            edge_list.append(edges)
            start += trigger_size
        edge_index = torch.cat(edge_list, dim=1)

        # to undirected
        row = torch.cat([edge_index[0], edge_index[1]])
        col = torch.cat([edge_index[1], edge_index[0]])
        edge_index = torch.stack([row, col])

        return edge_index

    def inject_trigger(self, idx_attach, features, edge_index, edge_weight):
        self.trojan_model.eval()
        idx_attach, features, edge_index, edge_weight = idx_attach.to(self.device), features.to(self.device), \
            edge_index.to(self.device), edge_weight.to(self.device)

        trojan_feat = self.trojan_model(features[idx_attach])
        trojan_feat = trojan_feat.view([-1, features.shape[1]])
        update_feat = torch.cat([features, trojan_feat])

        trojan_edge = self.get_trojan_edge(len(features), idx_attach, self.trigger_size).to(self.device)
        injected_row, injected_col = trojan_edge[0], trojan_edge[1]
        trojan_weights = torch.clamp_min(torch.cosine_similarity(update_feat[injected_row], update_feat[injected_col]), self.thrd)

        update_edge_weights = torch.cat([edge_weight, trojan_weights])
        update_edge_index = torch.cat([edge_index, trojan_edge], dim=1)
        return update_feat, update_edge_index, update_edge_weights

    def fit(self, features, edge_index, edge_weight, labels, idx_train, idx_attach, idx_unlabeled):
        features, edge_index, labels, idx_train, idx_attach, idx_unlabeled = features.to(self.device), edge_index.to(self.device), \
            labels.to(self.device), idx_train.to(self.device), idx_attach.to(self.device), idx_unlabeled.to(self.device)

        if self.trojan_model is None or self.shadow_model is None:
            self.init_models(features)

        if edge_weight is None:
            edge_weight = torch.ones([edge_index.shape[1]], device=self.device, dtype=torch.float)

        # get the trojan edges, which include the target-trigger edge and the edges among trigger using heuristic method
        trojan_edge = self.get_trojan_edge(len(features), idx_attach, self.trigger_size).to(self.device)

        optimizer_shadow = optim.Adam(self.shadow_model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        optimizer_trigger = optim.Adam(self.trojan_model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # change the labels of the poisoned node to the target class
        n_labels = labels.clone()
        n_labels[idx_attach] = self.target_class

        # update the poisoned graph's edge index
        poison_edge_index = torch.cat([edge_index, trojan_edge], dim=1)

        for i in range(self.outer_epochs):
            self.trojan_model.train()
            for j in range(self.inner_epochs):
                optimizer_shadow.zero_grad()
                optimizer_trigger.zero_grad()

                trojan_feat = self.trojan_model(features[idx_attach].to(self.device))
                trojan_feat = trojan_feat.view([-1, features.shape[1]])
                poison_x = torch.cat([features.to(self.device), trojan_feat])

                injected_row, injected_col = trojan_edge[0], trojan_edge[1]
                trojan_weights = torch.clamp_min(torch.cosine_similarity(poison_x[injected_row], poison_x[injected_col]), self.thrd)
                poison_edge_weights = torch.cat([edge_weight, trojan_weights])  # repeat trojan weights because of undirected edge

                output = self.shadow_model(poison_x, poison_edge_index, poison_edge_weights)
                loss_inner = F.cross_entropy(output[torch.cat([idx_train, idx_attach])],
                                             n_labels[torch.cat([idx_train, idx_attach])])  # add our adaptive loss

                loss_inner.backward()
                optimizer_shadow.step()
                optimizer_trigger.step()

    def get_poisoned(self, features, edge_index, edge_weight, labels, idx_attach):
        with torch.no_grad():
            poison_x, poison_edge_index, poison_edge_weights = self.inject_trigger(idx_attach, features, edge_index, edge_weight)
        poison_labels = labels.clone()
        poison_labels[idx_attach] = self.target_class
        return poison_x, poison_edge_index, poison_edge_weights, poison_labels
