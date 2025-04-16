from copy import deepcopy

import torch

import torch.autograd as autograd
from torch.nn.utils import parameters_to_vector


def euclidean_distance(ori_feat, target_feat):
    aa = ori_feat**2
    sum_aa = torch.sum(aa, dim=1).unsqueeze(1)

    bb = target_feat**2
    sum_bb = torch.sum(bb, dim=1).unsqueeze(0)
    return torch.sqrt(sum_aa + sum_bb - 2 * ori_feat.mm(target_feat.t()))


def construct_injected_graph(injected_feat, injected_edge_index, injected_edge_weight, ori_train_data):
    # Inject generated nodes into the original graph
    all_feat = torch.cat([ori_train_data.x, injected_feat], dim=0)
    all_edge_weight = torch.cat([ori_train_data.edge_weight, injected_edge_weight], dim=0)

    all_edge_index = torch.cat([ori_train_data.edge_index, injected_edge_index], dim=1)
    return all_feat, all_edge_index, all_edge_weight


class BEEF(object):

    def __init__(self, data, idx_attach, idx_attack, generator, n_round,
                 magnitude_thresh_ratio, balance_factor, generator_epochs, device):

        self.device = device
        self.generator = generator
        self.idx_attach, self.idx_attack = idx_attach.to(device), idx_attack.to(device)
        self.train_data, self.test_data = deepcopy(data).to(device), deepcopy(data).to(device)
        self.ori_train_data, self.ori_test_data = deepcopy(data).to(device), deepcopy(data).to(device)
        self.client_id, self.model, self.optimizer, self.local_epochs, self.loss_fn = None, None, None, None, None

        self.balance_factor = balance_factor
        self.n_round = n_round
        self.prev_model_vec = None
        self.prev_model_vec_list = []
        self.magnitude_thresh_ratio = magnitude_thresh_ratio
        self.generator_epochs = generator_epochs

    def initialize_attack(self, client_id, model, optimizer, local_epochs, loss_fn):
        self.model, self.optimizer = model, optimizer
        self.local_epochs = local_epochs
        self.loss_fn = loss_fn
        self.client_id = client_id

    def solely_local_model_training(self):
        # Reset training settings
        self.optimizer.param_groups.clear()
        self.optimizer.state.clear()
        self.optimizer.add_param_group({'params': self.model.parameters()})

        ori_train_data, ori_test_data = deepcopy(self.ori_train_data), deepcopy(self.ori_test_data)

        # Local training on benign samples
        self.model.train()
        for idx in range(self.local_epochs):
            self.optimizer.zero_grad()

            feat, edge_index, edge_weight = ori_train_data.x, ori_train_data.edge_index, ori_train_data.edge_weight
            output = self.model(feat, edge_index, edge_weight)
            train_mask = deepcopy(self.ori_train_data.train_mask)
            train_mask[self.idx_attach] = False
            train_idx = torch.arange(train_mask.size(0), device=self.device, dtype=torch.long)[train_mask]
            loss_train = self.loss_fn(output[train_idx], ori_train_data.y[train_idx])
            loss_train.backward()
            self.optimizer.step()

            print('\tBEEF: Client ID = {}@Normal training@Classification loss = {:.3f}'.format(
                self.client_id, loss_train
            ))

            pred = output[train_idx].max(1)[1]

    def solely_trigger_generator_training(self):
        # Reset training settings
        self.generator.optimizer.param_groups.clear()
        self.generator.optimizer.state.clear()
        self.generator.optimizer.add_param_group({'params': self.generator.trigger_model.parameters()})

        ori_train_data, ori_test_data = deepcopy(self.ori_train_data), deepcopy(self.ori_test_data)
        poisoned_labels = ori_train_data.y.clone()
        poisoned_labels[self.idx_attach] = self.generator.target_class

        # Trigger generator training on malicious samples
        self.model.eval()
        self.generator.trigger_model.train()
        for idx in range(self.generator_epochs):
            self.generator.optimizer.zero_grad()

            # Training Trigger Model
            injected_feat, injected_edge_index, injected_edge_weight = self.generator.trigger_model(
                ori_train_data.x, ori_train_data.edge_index, self.idx_attach
            )
            all_feat, all_edge_index, all_edge_weight = construct_injected_graph(
                injected_feat, injected_edge_index, injected_edge_weight, ori_train_data,
            )

            # Calculate classification loss
            output = self.model(all_feat, all_edge_index, all_edge_weight)
            loss_clf = self.loss_fn(output[self.idx_attach], poisoned_labels[self.idx_attach])

            # regularize loss
            injected_row, injected_col = injected_edge_index[0], injected_edge_index[1]
            sim_distance = 0.5 - 0.5 * torch.cosine_similarity(all_feat[injected_row], all_feat[injected_col])
            loss_reg = torch.mean(sim_distance)

            loss = loss_clf + self.balance_factor * loss_reg
            print('\tBEEF: Client ID = {}@Generator Training@Total loss = {:.3f}@Classification loss = {:.3f}@Regularize loss = {:.3f}'.format(
                self.client_id, loss.item(), loss_clf.item(), loss_reg.item()
            ))

            loss.backward()

            self.generator.optimizer.step()

    def collaborative_training(self, flr, cur_grad_magnitude_mask):
        # Reset training settings
        self.optimizer.param_groups.clear()
        self.optimizer.state.clear()
        self.optimizer.add_param_group({'params': self.model.parameters()})

        self.generator.optimizer.param_groups.clear()
        self.generator.optimizer.state.clear()
        self.generator.optimizer.add_param_group({'params': self.generator.trigger_model.parameters()})

        ori_train_data, ori_test_data = deepcopy(self.ori_train_data), deepcopy(self.ori_test_data)
        poisoned_labels = ori_train_data.y.clone()
        poisoned_labels[self.idx_attach] = self.generator.target_class

        # Simultaneously train trigger generator and poisoned models
        cur_grad_mask = None
        self.model.train()
        self.generator.trigger_model.train()
        for idx in range(self.local_epochs):
            self.model.zero_grad()
            self.generator.optimizer.zero_grad()

            # Training Trigger Model
            injected_feat, injected_edge_index, injected_edge_weight = self.generator.trigger_model(
                ori_train_data.x, ori_train_data.edge_index, self.idx_attach
            )
            all_feat, all_edge_index, all_edge_weight = construct_injected_graph(
                injected_feat, injected_edge_index, injected_edge_weight, ori_train_data,
            )

            # Getting locations of neurons with positive influences on predictions
            output = self.model(all_feat, all_edge_index, all_edge_weight)
            poisoned_logit = output[self.idx_attach][None, self.generator.target_class].mean()
            grads = autograd.grad(poisoned_logit, self.model.parameters(), retain_graph=False, create_graph=False)
            malicious_importance = parameters_to_vector(grads)

            malicious_pos_importance_mask = malicious_importance > 0.
            if torch.sum(malicious_pos_importance_mask.float()) == 0:
                malicious_importance_grad = malicious_importance > 0.
                cur_grad_mask = cur_grad_magnitude_mask & malicious_importance_grad
            else:
                magnitude_thresh = torch.quantile(
                    malicious_importance[malicious_pos_importance_mask].reshape(-1), q=(1 - self.magnitude_thresh_ratio)
                ).item()
                malicious_importance_grad = malicious_importance >= magnitude_thresh
                cur_grad_mask = cur_grad_magnitude_mask & malicious_importance_grad
                print('\tBEEF: Thresh = {:.4f}@The number of gradient mask = {}'.format(
                    magnitude_thresh, torch.sum(cur_grad_mask.int()).item()
                ))

            # Training Local Model
            output = self.model(all_feat, all_edge_index, all_edge_weight)
            node_idx = torch.arange(ori_train_data.train_mask.size(0), device=self.device, dtype=torch.long)
            train_idx = node_idx[ori_train_data.train_mask]
            loss_train = self.loss_fn(output[train_idx], poisoned_labels[train_idx])
            loss_train.backward()

            # Changing gradients of victim model's parameters
            pointer = 0  # Pointer for slicing the vector for each parameter
            for name, param in self.model.named_parameters():
                # The length of the parameter
                param_len = param.numel()

                # Slice the vector, reshape it, and replace the old data of the parameter
                param_grad_mask = cur_grad_mask[pointer: pointer + param_len].view_as(param)
                param.grad = param.grad * param_grad_mask.float()

                # Increment the pointer
                pointer += param_len

            self.generator.optimizer.step()
            self.optimizer.step()

            print('\tBEEF: Client ID = {} -- Round: {} -- Client: {} -- Epoch: {} -- Loss: {:.4f} -- Malicious: {}'.format(
                self.client_id, flr, self.client_id, idx, loss_train.item(), 'True'
            ))

    def run_exec(self, flr, server_model_state_dict, **kwargs):
        self.model.load_state_dict(server_model_state_dict)
        cur_model_vec = deepcopy(parameters_to_vector(self.model.parameters()).detach())

        # Calculate the mask of updates of model parameters
        if self.prev_model_vec is None:
            self.prev_model_vec = parameters_to_vector(self.model.parameters()).detach()

        self.prev_model_vec_list.append(torch.abs(cur_model_vec - self.prev_model_vec))
        if len(self.prev_model_vec_list) > self.n_round:
            self.prev_model_vec_list.pop(0)

        magnitude_thresh = 0
        for model_diff in self.prev_model_vec_list:
            magnitude_thresh += torch.quantile(model_diff.reshape(-1), q=self.magnitude_thresh_ratio).item()
        magnitude_thresh /= len(self.prev_model_vec_list)

        cur_grad_magnitude_mask = torch.abs(cur_model_vec - self.prev_model_vec) <= magnitude_thresh
        print('\tBEEF: Client ID = {}@The number of magnitude mask = {}'.format(
            self.client_id, torch.sum(cur_grad_magnitude_mask.int()).item())
        )

        self.generator.trigger_model.tau *= 0.999
        self.solely_local_model_training()
        self.solely_trigger_generator_training()
        self.collaborative_training(flr, cur_grad_magnitude_mask)

        ori_train_data, ori_test_data = deepcopy(self.ori_train_data), deepcopy(self.ori_test_data)
        self.train_data, self.test_data = self.generator.poison_train_graph(ori_train_data, self.idx_attach), \
            self.generator.poison_test_graph(ori_test_data, self.idx_attack)

        # Reserving the current model's parameters
        self.prev_model_vec = deepcopy(parameters_to_vector(self.model.parameters()).detach())

        return self.model

    def gen_poisoned_data(self):
        ori_train_data, ori_test_data = deepcopy(self.ori_train_data), deepcopy(self.ori_test_data)
        self.train_data, self.test_data = self.generator.poison_train_graph(ori_train_data, self.idx_attach), \
            self.generator.poison_test_graph(ori_test_data, self.idx_attack)
