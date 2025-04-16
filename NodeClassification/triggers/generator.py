
import torch

from triggers.gta import GTABackdoor
from triggers.heuristic import HeuristicBackdoor
from triggers.ugba import UGBABackdoor

from triggers.position import obtain_attach_nodes, cluster_distance_selection, \
    cluster_degree_selection, obtain_attach_nodes_degree, obtain_attach_nodes_cluster


class TriggerGenerator(object):

    def __init__(self, trigger_type, trigger_position, poisoning_intensity, dataset_name, num_classes, target_class, trigger_size, thrd,
                 target_loss_weight, homo_loss_weight, dd_loss_weight, homo_boost_thrd, density, degree, dis_weight, num_hidden, dropout,
                 train_epochs, lr, weight_decay, outer_epochs, inner_epochs, seed, device):
        self.trigger_type, self.trigger_position = trigger_type, trigger_position

        self.num_poisoned_nodes = 0
        self.poisoning_density = poisoning_intensity
        if self.trigger_type == 'gta':
            self.trigger_model = GTABackdoor(
                target_class, num_hidden, thrd, trigger_size, lr, weight_decay, outer_epochs, inner_epochs, num_classes, device
            )
        elif self.trigger_type == 'ugba':
            self.trigger_model = UGBABackdoor(
                target_class, num_hidden, thrd, trigger_size, lr, weight_decay,
                outer_epochs, inner_epochs, target_loss_weight, homo_loss_weight, homo_boost_thrd, num_classes, device
            )
        elif self.trigger_type in ['renyi', 'ws', 'ba']:
            self.trigger_model = HeuristicBackdoor(
                target_class, num_hidden, thrd, trigger_size, trigger_type, density, degree, lr, weight_decay, outer_epochs,
                inner_epochs, num_classes, device
            )
        else:
            raise NotImplementedError('Trigger Type {} is Not Defined!'.format(self.trigger_type))
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

    def train_generator(self, train_data, idx_train, idx_attach, idx_unlabelled):
        self.trigger_model.fit(
            train_data.x, train_data.edge_index,
            train_data.edge_weight, train_data.y, idx_train, idx_attach, idx_unlabelled
        )

    def poison_train_graph(self, train_data, idx_attach):
        train_data.x, train_data.edge_index, train_data.edge_weight, train_data.y = self.trigger_model.get_poisoned(
            train_data.x, train_data.edge_index, train_data.edge_weight, train_data.y, idx_attach
        )

        train_mask = torch.zeros_like(train_data.train_mask, dtype=torch.bool)
        train_mask[idx_attach] = True
        train_data.train_mask = torch.bitwise_or(train_data.train_mask, train_mask)

        return train_data

    def poison_test_graph(self, test_data, idx_attack):
        test_data.x, test_data.edge_index, test_data.edge_weight, test_data.y = self.trigger_model.get_poisoned(
            test_data.x, test_data.edge_index, test_data.edge_weight, test_data.y, idx_attack
        )

        test_mask = torch.zeros_like(test_data.test_mask, dtype=torch.bool)
        test_mask[idx_attack] = True
        test_data.test_mask = torch.bitwise_or(test_data.test_mask, test_mask)
        return test_data
