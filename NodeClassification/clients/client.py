""" This module contains the client class for the federated learning system. """
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import parameters_to_vector

from torch_geometric.utils import to_undirected

from attacks.beef import BEEF
from clients.utils import local_data_split, subgraph
from utils import eval_model


class FLClient(object):

    def __init__(self, client_id, is_malicious, data, generator, attack_name, aggr_name,
                 ratio_train, ratio_val, ratio_test, model, lr, weight_decay, local_epochs, l1, loc_l2,
                 eps, scale, mu, magnitude_thresh_ratio, alpha, n_round, balance_factor, generator_epochs, adv_local_epochs, device):

        self.lr = lr
        self.client_id = client_id
        self.weight_decay = weight_decay
        self.model = deepcopy(model)
        self.is_malicious = is_malicious
        self.l1 = l1
        self.loc_l2 = loc_l2

        self.device = device
        self.local_epochs = local_epochs
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        self.ratio_train, self.ratio_val, self.ratio_test = ratio_train, ratio_val, ratio_test
        self.data, self.idx_train, self.idx_val, self.idx_clean_test, self.idx_unlabeled, self.idx_attack = data, None, None, None, None, None
        self.local_dataset()

        self.model = self.model.to(device)
        if self.data is not None:
            self.data = self.data.to(device)

        self.attack_name = attack_name
        self.generator = generator
        self.idx_attach = self.generator.gen_idx_attach(self.data, self.data.edge_index, self.idx_train, self.idx_val, self.idx_clean_test, self.idx_unlabeled)

        self.adv_local_epochs = adv_local_epochs

        # Training and Testing mask
        if self.idx_attach is not None:
            self.idx_attach = self.idx_attach.to(self.device)
            train_mask = torch.zeros_like(self.data.train_mask, dtype=torch.bool, device=self.device)
            train_mask[self.idx_attach] = True
            self.data.train_mask = torch.bitwise_or(self.data.train_mask, train_mask)

        if self.idx_attack is not None:
            self.idx_attack = self.idx_attack.to(self.device)
            test_mask = torch.zeros_like(self.data.test_mask, dtype=torch.bool, device=self.device)
            test_mask[self.idx_attack] = True
            self.data.test_mask = torch.bitwise_or(self.data.test_mask, test_mask)

        if is_malicious and attack_name == 'beef':
            self.attacker = BEEF(
                self.data, self.idx_attach, self.idx_attack, self.generator,
                n_round, magnitude_thresh_ratio=magnitude_thresh_ratio, device=self.device,
                balance_factor=balance_factor, generator_epochs=generator_epochs
            )

        self.train_data, self.test_data = self.data, self.data

        self.poison_test_data = None
        if self.is_malicious and self.attack_name != 'beef':
            self.poison_test_data = deepcopy(self.data)
            self.poison_test_data = self.generator.poison_test_graph(self.poison_test_data, self.idx_attack)

        self.mu = mu
        self.aggr_name = aggr_name

    def local_dataset(self):
        self.data, self.idx_train, self.idx_val, self.idx_clean_test, self.idx_attack = local_data_split(
            self.data, self.ratio_train, self.ratio_val, self.ratio_test
        )
        edge_weight = torch.ones(self.data.edge_index.shape[1], device=self.device, dtype=torch.float32)
        self.data.edge_weight = edge_weight

        self.data.edge_index = to_undirected(self.data.edge_index)
        train_edge_index, _, edge_mask = subgraph(torch.bitwise_not(self.data.test_mask), self.data.edge_index)
        mask_edge_index = self.data.edge_index[:, torch.bitwise_not(edge_mask)]
        idx_unlabeled = (torch.bitwise_not(self.data.test_mask) & torch.bitwise_not(self.data.train_mask)).nonzero().flatten()
        self.data.mask_edge_index = mask_edge_index
        self.idx_unlabeled = idx_unlabeled

    def init_poisoned_graph(self):
        self.poison_test_data = deepcopy(self.data)
        self.idx_attack = self.idx_attack.to(self.device)
        self.poison_test_data = self.generator.poison_test_graph(self.poison_test_data, self.idx_attack)

    def run_exec(self, flr, server_model_state_dict, conducted_attack=True):

        # Reset training settings
        self.model.load_state_dict(server_model_state_dict)
        self.optimizer.param_groups.clear()
        self.optimizer.state.clear()
        self.optimizer.add_param_group({'params': self.model.parameters()})

        if self.is_malicious and conducted_attack and isinstance(self.attacker, BEEF):
            self.attacker.initialize_attack(self.client_id, self.model, self.optimizer, self.local_epochs, self.loss_fn)

        self.model.train()

        origin_vec, origin_state_dict = None, None
        if self.is_malicious and conducted_attack and self.attack_name in ['beef']:
            self.model = self.attacker.run_exec(flr, server_model_state_dict)
        else:
            self.data = self.data.to(self.device)
            for idx in range(self.local_epochs):
                self.optimizer.zero_grad()

                output = self.model(self.data.x, self.data.edge_index, self.data.edge_weight)
                node_idx = torch.arange(self.data.train_mask.size(0), device=self.device, dtype=torch.long)
                train_idx = node_idx[self.data.train_mask]
                loss_train = self.loss_fn(output[train_idx], self.data.y[self.data.train_mask])

                loss_train.backward()
                self.optimizer.step()
                print('Round: {} -- Client: {} -- Epoch: {} -- Loss: {:.4f} -- Malicious: {}'.format(
                    flr, self.client_id, idx, loss_train.item(), self.is_malicious and conducted_attack
                ))

        self.data = self.data.to(self.device)
        acc_clean_test = eval_model(self.model, self.data, self.idx_clean_test, self.device)
        print('Round: {} -- Client: {} -- Clean Test Accuracy: {:.4f}'.format(flr, self.client_id, acc_clean_test))

        if self.is_malicious and conducted_attack:
            self.attacker.test_data = self.attacker.test_data.to(self.device)
            acc_poison_test = eval_model(self.model, self.attacker.test_data, self.idx_attack, self.device)
            print('Round: {} -- Client: {} -- Poisoned Test Accuracy: {:.4f}'.format(flr, self.client_id, acc_poison_test))

        if self.is_malicious and self.attack_name == 'beef':
            self.attacker.gen_poisoned_data()
            self.poison_test_data = deepcopy(self.attacker.test_data)

        return deepcopy(self.model.state_dict())
