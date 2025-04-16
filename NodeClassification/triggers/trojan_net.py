import copy
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class GradWhere(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input, thrd, device):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        rst = torch.where(input > thrd, torch.tensor(1.0, device=device, requires_grad=True),
                          torch.tensor(0.0, device=device, requires_grad=True))
        return rst

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()

        """
        Return results number should corresponding with .forward inputs (besides ctx),
        for each input, return a corresponding backward grad
        """
        return grad_input, None, None


class GraphTrojanNet(nn.Module):
    # In the future, we may use a GNN model to generate backdoor
    def __init__(self, n_feat, hidden_dim, trigger_size, tau, layer_num=1, dropout=0.00):
        super(GraphTrojanNet, self).__init__()
        self.tau = tau

        layers = []
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        for l in range(layer_num - 1):
            layers.append(nn.Linear(n_feat, n_feat))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))

        self.layers = nn.Sequential(*layers)

        self.feat = nn.Linear(n_feat, trigger_size * n_feat)
        self.edge = nn.Linear(n_feat, int(trigger_size * (trigger_size - 1) / 2))
        self.trigger_size = trigger_size
        self.trigger_index = self.get_trigger_index(trigger_size)
        print(self.trigger_index)

    def get_trigger_index(self, trigger_size):
        edge_list = [[0, 0]]
        for j in range(trigger_size):
            for k in range(j):
                edge_list.append([j, k])
        edge_index = torch.tensor(edge_list).long().T
        return edge_index

    def get_trojan_edge(self, start, target_idx):
        edge_list = []
        for idx in target_idx:
            edges = self.trigger_index.clone()
            edges[0, 0] = idx
            edges[1, 0] = start
            edges[:, 1:] = edges[:, 1:] + start

            edge_list.append(edges)
            start += self.trigger_size
        edge_index = torch.cat(edge_list, dim=1)
        # to undirected
        row = torch.cat([edge_index[0], edge_index[1]])
        col = torch.cat([edge_index[1], edge_index[0]])
        edge_index = torch.stack([row, col])
        return edge_index

    def forward(self, feat, edge_index, target_idx):

        """
        "input", "mask" and "thrd", should already in cuda before sent to this function.
        If using sparse format, corresponding tensor should already in sparse format before
        sent into this function
        """

        GW = GradWhere.apply
        feat_list, edge_weight_list = [], []

        for node_id in target_idx:

            h = self.layers(feat[node_id])

            val_min, val_max = torch.min(feat).item(), torch.max(feat).item()
            if val_max <= 1.0 and val_min >= 0.:
                n_feat = torch.sigmoid(self.feat(h))
            else:
                # OGBN-arXiv范围不是0-1
                n_feat = self.feat(h)
                # feat = self.feat(h) - 2.0

            edge_weight = self.edge(h)
            edge_weight = GW(edge_weight, self.tau, feat.device)

            n_feat = n_feat.reshape(-1, feat.shape[-1])

            feat_list.append(n_feat)
            edge_weight_list.append(edge_weight.flatten())
            
        
        n_feat = torch.cat(feat_list, dim=0)
        n_edge_weight = torch.cat(edge_weight_list, dim=0)
        n_edge_weight = torch.cat([torch.ones((len(target_idx),), dtype=torch.float, device=feat.device), n_edge_weight], dim=0)
        n_edge_weight = torch.cat([n_edge_weight, n_edge_weight], dim=0)
        n_edge_index = self.get_trojan_edge(feat.shape[0], target_idx).to(feat.device)
        return n_feat, n_edge_index, n_edge_weight