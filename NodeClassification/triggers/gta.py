import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from models.gcn import GCN


class GradWhere(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """
    @staticmethod
    def forward(ctx, inputs, thrd):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(inputs)
        rst = torch.where(
            inputs > thrd,
            torch.tensor(1.0, device=inputs.device, requires_grad=True), torch.tensor(0.0, device=inputs.device, requires_grad=True)
        )
        return rst

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        inputs, = ctx.saved_tensors
        grad_input = grad_output.clone()

        """
        Return results number should corresponding with .forward inputs (besides ctx),
        for each input, return a corresponding backward grad
        """
        return grad_input, None, None


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

        self.edge = nn.Linear(num_feat, int(num_out * (num_out - 1) / 2))

    def forward(self, inputs, thrd):
        """
        "input", "mask" and "thrd", should already in cuda before sent to this function.
        If using sparse format, corresponding tensor should already in sparse format before
        sent into this function
        """
        GW = GradWhere.apply
        h = self.layers(inputs)

        feat = self.feat(h)
        edge_weight = self.edge(h)

        edge_weight = GW(edge_weight, thrd)
        return feat, edge_weight


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


class GTABackdoor:

    def __init__(self, target_class, num_hidden, thrd, trigger_size, lr, weight_decay, outer_epochs, inner_epochs, num_classes, device):
        self.device = device
        self.thrd = thrd
        self.num_hidden = num_hidden
        self.target_class = target_class
        self.trigger_size = trigger_size
        self.lr, self.weight_decay = lr, weight_decay
        self.trigger_index = self.get_trigger_index(trigger_size)
        self.inner_epochs, self.outer_epochs = inner_epochs, outer_epochs
        self.trojan_model, self.shadow_model, self.num_classes = None, None, num_classes

    def init_models(self, features, labels):
        # initialize a shadow model
        self.shadow_model = GCN(
            num_feat=features.shape[1], num_hidden=self.num_hidden, num_classes=self.num_classes, dropout=0.0
        ).to(self.device)

        # initialize a trojanNet to generate trigger
        self.trojan_model = GraphTrojanNet(num_feat=features.shape[1], num_out=self.trigger_size, num_hidden=self.num_hidden, num_layers=2).to(self.device)

    def get_trigger_index(self, trigger_size):
        edge_list = [[0, 0]]
        for j in range(trigger_size):
            for k in range(j):
                edge_list.append([j, k])
        edge_index = torch.tensor(edge_list, device=self.device).long().T

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

        trojan_feat, trojan_weights = self.trojan_model(features[idx_attach], self.thrd)
        trojan_weights = torch.cat([torch.ones([len(idx_attach), 1], dtype=torch.float, device=self.device), trojan_weights], dim=1)
        trojan_weights = trojan_weights.flatten()

        trojan_feat = trojan_feat.view([-1, features.shape[1]])
        trojan_edge = self.get_trojan_edge(len(features), idx_attach, self.trigger_size).to(self.device)

        update_edge_weights = torch.cat([edge_weight, trojan_weights, trojan_weights])
        update_feat = torch.cat([features, trojan_feat])
        update_edge_index = torch.cat([edge_index, trojan_edge], dim=1)
        return update_feat, update_edge_index, update_edge_weights

    def fit(self, features, edge_index, edge_weight, labels, idx_train, idx_attach, idx_unlabeled):
        features, edge_index, labels, idx_train, idx_attach, idx_unlabeled = features.to(self.device), edge_index.to(self.device), \
            labels.to(self.device), idx_train.to(self.device), idx_attach.to(self.device), idx_unlabeled.to(self.device)

        if self.trojan_model is None or self.shadow_model is None:
            self.init_models(features, labels)

        if edge_weight is None:
            edge_weight = torch.ones([edge_index.shape[1]], device=self.device, dtype=torch.float)

        optimizer_shadow = optim.Adam(self.shadow_model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        optimizer_trigger = optim.Adam(self.trojan_model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # change the labels of the poisoned node to the target class
        n_labels = labels.clone()
        n_labels[idx_attach] = self.target_class

        # get the trojan edges, which include the target-trigger edge and the edges among trigger
        trojan_edge = self.get_trojan_edge(len(features), idx_attach, self.trigger_size).to(self.device)

        # update the poisoned graph's edge index
        poison_edge_index = torch.cat([edge_index, trojan_edge], dim=1)

        # further change it to bi-level optimization
        for i in range(self.outer_epochs):
            self.trojan_model.train()
            for j in range(self.inner_epochs):
                optimizer_shadow.zero_grad()
                optimizer_trigger.zero_grad()
                trojan_feat, trojan_weights = self.trojan_model(features[idx_attach].to(self.device), self.thrd)
                trojan_weights = torch.cat([torch.ones([len(trojan_feat), 1], dtype=torch.float, device=self.device), trojan_weights], dim=1)
                trojan_weights = trojan_weights.flatten()
                trojan_feat = trojan_feat.view([-1, features.shape[1]])
                poison_edge_weights = torch.cat([edge_weight, trojan_weights, trojan_weights])  # repeat trojan weights because of undirected edge
                poison_x = torch.cat([features.to(self.device), trojan_feat])

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
        poison_edge_index = poison_edge_index[:, poison_edge_weights > 0.0]
        poison_edge_weights = poison_edge_weights[poison_edge_weights > 0.0]

        return poison_x, poison_edge_index, poison_edge_weights, poison_labels
