"""FL Server"""
import time
import numpy as np

from copy import deepcopy

import torch
from torch_geometric.utils import to_undirected

from clients.client import FLClient
from clients.utils import local_data_split, subgraph
from dataset.utils import dataset_distribution
from models.utils import model_construct
from defenses.fedavg import FedAvg
from triggers.gen_trigger import GenTrigger
from utils import eval_model
from triggers.generator import TriggerGenerator


class FLServer(object):
    ignored_weights = ['num_batches_tracked']

    def __init__(self, num_clients, num_malicious, num_clients_per_round, dataset, data_distribution, non_iid_degree, dataset_name,
                 model_state_dict, model_name, attack_name, aggr_name, trigger_type, trigger_position, poisoning_intensity, target_class,
                 trigger_size, thrd, ratio_train, ratio_val, ratio_test, target_loss_weight, homo_loss_weight, dd_loss_weight, homo_boost_thrd,
                 density, degree, dis_weight, train_epochs, outer_epochs, inner_epochs, lr, weight_decay, local_epochs, adv_local_epochs, num_hidden, num_classes,
                 dropout, eps, scale, seed, device, fl_rounds, mu, backdoor_start_round, backdoor_end_round, num_hidden_generator, magnitude_thresh_ratio,
                 tau, alpha, n_round, balance_factor, generator_epochs, lamda, l1, loc_l2, aggr_norm, norm_scale, auror_alpha, auror_tau, rlr_theta, rlr_lr, logger):

        # Splitting Dataset ```The last-one dataset is applied to train the trigger generator```
        self.data_distribution = data_distribution
        self.graphs = dataset_distribution(dataset_name, dataset, num_clients + 1, data_distribution, non_iid_degree, target_class)
        self.device = device

        # Initialize base settings
        self.fl_rounds = fl_rounds
        self.backdoor_start_round = backdoor_start_round
        self.backdoor_end_round = backdoor_end_round
        self.aggr_name = aggr_name
        self.adv_local_epochs = adv_local_epochs

        # Initialize Models
        masked = False
        if self.aggr_name == 'fedpub':
            # I don't know, masking is not working
            masked = False
        self.model = model_construct(dataset, num_hidden, num_classes, dropout, model_name, masked=masked, l1=l1).to(self.device)
        if hasattr(self.model, 'initialize'):
            self.model.initialize()
        if model_state_dict is not None:
            self.model.load_state_dict(model_state_dict)

        # Initialize malicious clients
        self.num_malicious = num_malicious
        self.malicious_clients = np.random.choice(num_clients, num_malicious, replace=False).tolist()

        # Initialize trigger generations
        if trigger_type != 'beef':
            data, idx_train, idx_val, idx_clean_test, _, idx_unlabeled = self.local_dataset(
                self.graphs[-1], ratio_train, ratio_val, ratio_test
            )
            self.generator = TriggerGenerator(
                trigger_type, trigger_position, poisoning_intensity, dataset_name, num_classes, target_class, trigger_size, thrd,
                target_loss_weight, homo_loss_weight, dd_loss_weight, homo_boost_thrd, density, degree, dis_weight, num_hidden, dropout, train_epochs,
                lr, weight_decay, outer_epochs, inner_epochs, seed, device
            )
            idx_attach = self.generator.gen_idx_attach(data, data.edge_index, idx_train, idx_val, idx_clean_test, idx_unlabeled)
            self.generator.train_generator(data, idx_train, idx_attach, idx_unlabeled)
        else:
            self.generator = GenTrigger(
                trigger_size, trigger_position, poisoning_intensity, dataset.x.shape[1], dataset_name, num_classes, target_class,
                num_hidden_generator, dis_weight, num_hidden, dropout, train_epochs, lr, weight_decay, tau, seed, device
            )

        # Initialize clients
        self.client_list = []
        self.num_clients = num_clients
        self.num_clients_per_round = num_clients_per_round
        for client_id in range(num_clients):
            is_malicious = client_id in self.malicious_clients
            print('\n\nInitialize client {}\'s dataset'.format(client_id))
            epochs = local_epochs

            self.client_list.append(
                FLClient(
                    client_id, is_malicious, self.graphs[client_id], self.generator, attack_name, self.aggr_name,
                    ratio_train, ratio_val, ratio_test, self.model, lr, weight_decay, epochs, l1, loc_l2, eps, scale,
                    mu, magnitude_thresh_ratio, alpha, n_round, balance_factor, generator_epochs, adv_local_epochs, device
                )
            )

        print('All clients initialized')

        self.aggr_method = None
        if self.aggr_name == 'fedavg':
            self.aggr_method = FedAvg(device=device)
        self.logger = logger

    def local_dataset(self, data, ratio_train, ratio_val, ratio_test):
        data, idx_train, idx_val, idx_clean_test, idx_attack = local_data_split(
            data, ratio_train, ratio_val, ratio_test
        )
        edge_weight = torch.ones(data.edge_index.shape[1], device=self.device, dtype=torch.float32)
        data.edge_weight = edge_weight

        data.edge_index = to_undirected(data.edge_index)
        train_edge_index, _, edge_mask = subgraph(torch.bitwise_not(data.test_mask), data.edge_index)
        mask_edge_index = data.edge_index[:, torch.bitwise_not(edge_mask)]
        idx_unlabeled = (torch.bitwise_not(data.test_mask) & torch.bitwise_not(data.train_mask)).nonzero().flatten()
        data.mask_edge_index = mask_edge_index
        return data, idx_train, idx_val, idx_clean_test, idx_attack, idx_unlabeled

    def check_ignored_weights(self, name) -> bool:
        for ignored in self.ignored_weights:
            if ignored in name:
                return True
        return False

    def update_global_model(self, weight_accumulator, global_model):
        for name, sum_update in weight_accumulator.items():
            if self.check_ignored_weights(name):
                continue
            model_weight = global_model.state_dict()[name]
            model_weight.add_(sum_update)

    def exec(self):
        # Conduct FL round
        for flr in range(self.fl_rounds):
            chosen_clients = np.random.choice(self.num_clients, self.num_clients_per_round,  replace=False).tolist()
            chosen_clients = sorted(chosen_clients)
            print('\n\nRound: {} -- Chosen Clients: {}'.format(flr, chosen_clients))

            client_local_state_dict, client_local_embeddings = [], []
            if self.aggr_name == 'fedpub':
                # pFL
                prev_model_state_dict = []
                for client_id in chosen_clients:
                    prev_model_state_dict.append(deepcopy(self.client_list[client_id].model.state_dict()))
            else:
                prev_model_state_dict = deepcopy(self.model.state_dict())
            for client_id in chosen_clients:
                begin_time = time.time()
                if self.aggr_name == 'fedpub':
                    state_dict = prev_model_state_dict[client_id]
                else:
                    state_dict = prev_model_state_dict

                conducted_attack = (client_id in self.malicious_clients) and not (flr < self.backdoor_start_round) and not (flr > self.backdoor_end_round)
                if self.aggr_name == 'fedpub':
                    local_state_dict, local_embeddings = self.client_list[client_id].run_exec(flr, state_dict, conducted_attack)
                    client_local_embeddings.append(local_embeddings)
                else:
                    local_state_dict = self.client_list[client_id].run_exec(flr, state_dict, conducted_attack)
                client_local_state_dict.append(local_state_dict)
                end_time = time.time()
                print('Client ID = {}@Duration Time = {:.4f}s'.format(client_id, end_time - begin_time))

            if self.aggr_name == 'fedpub':
                global_weights, client2local_weights = self.aggr_method.aggr(prev_model_state_dict,
                                                                             client_local_state_dict, client_local_embeddings, flr=flr)
                self.model.load_state_dict(global_weights)
                for cid in range(self.num_clients):
                    self.client_list[cid].model.load_state_dict(client2local_weights[cid])
            else:
                weight_accumulator = self.aggr_method.aggr(prev_model_state_dict, client_local_state_dict, flr=flr, global_model=self.model)
                self.update_global_model(weight_accumulator, self.model)

            clean_acc, malicious_asr, benign_asr = [], [], []
            for client_id in range(self.num_clients):

                # Average Clean ACC
                client = self.client_list[client_id]
                if self.aggr_name == 'fedpub':
                    if self.data_distribution in ['oneclass', 'twoclass', 'inter-client']:
                        acc = eval_model(self.model, client.data, client.idx_clean_test, self.device)
                    else:
                        acc = eval_model(client.model, client.data, client.idx_clean_test, self.device)
                else:
                    acc = eval_model(self.model, client.data, client.idx_clean_test, self.device)
                clean_acc.append(acc)

                # Average ASR
                if client_id in self.malicious_clients and flr >= self.backdoor_start_round:
                    if self.aggr_name == 'fedpub':
                        if self.data_distribution in ['oneclass', 'twoclass', 'inter-client']:
                            asr = eval_model(self.model, client.test_data, client.idx_attack, self.device)
                        else:
                            asr = eval_model(self.client_list[client_id].model, client.test_data, client.idx_attack, self.device)
                    else:
                        asr = eval_model(self.model, client.test_data, client.idx_attack, self.device)
                    malicious_asr.append(asr)

                # Average ASR on benign clients
                if client_id not in self.malicious_clients and self.num_malicious > 0 and flr >= self.backdoor_start_round:
                    self.client_list[client_id].init_poisoned_graph()

                    if self.aggr_name == 'fedpub':
                        if self.data_distribution in ['oneclass', 'twoclass', 'inter-client']:
                            asr = eval_model(self.model,
                                             self.client_list[client_id].poison_test_data, self.client_list[client_id].idx_attack, self.device)
                        else:
                            asr = eval_model(self.client_list[client_id].model,
                                             self.client_list[client_id].poison_test_data, self.client_list[client_id].idx_attack, self.device)
                    else:
                        asr = eval_model(self.model,
                                         self.client_list[client_id].poison_test_data, self.client_list[client_id].idx_attack, self.device)
                    benign_asr.append(asr)

            clean_acc, malicious_asr, benign_asr = np.mean(clean_acc), np.mean(malicious_asr) if len(malicious_asr) > 0 else 0, np.mean(benign_asr) if len(malicious_asr) > 0 else 0
            if self.logger is not None:
                self.logger.log({
                    'clean_acc': clean_acc,
                    'malicious_asr': malicious_asr,
                    'benign_asr': benign_asr,
                })
            print('Round: {} -- Clean ACC: {:.4f} -- Malicious ASR: {:.4f} -- Benign ASR: {:.4f}'.format(flr, clean_acc, malicious_asr, benign_asr))

        return self.model
