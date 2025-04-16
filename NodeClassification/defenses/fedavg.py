import math

import torch


class FedAvg:

    ignored_weights = ['num_batches_tracked']

    def __init__(self, device) -> None:
        self.device = device

    # FedAvg aggregation
    def aggr(self, server_state_dict, clients_state_dict, *args, **kwargs):
        weight_accumulator = {}
        for local_state_dict in clients_state_dict:
            client_update = {}
            for param_name, value in local_state_dict.items():
                if self.check_ignored_weights(param_name):
                    continue
                client_update[param_name] = value - server_state_dict[param_name]
            self.accumulate_weights(weight_accumulator, client_update, scale=len(clients_state_dict))

        # # Assessing Similarities between normal clients and malicious clients
        # import numpy as np
        # import sklearn.metrics.pairwise as smp
        #
        # local_vec_params = []
        # num_client_models = len(clients_state_dict)
        # for local_state_dict in clients_state_dict:
        #     local_vec_params.append(self.get_vectorized_params(local_state_dict).cpu().numpy())
        # local_vec_params = np.array(local_vec_params, dtype=np.double)
        # cd = smp.cosine_distances(local_vec_params.reshape(num_client_models, -1))
        # for i in range(num_client_models):
        #     for j in range(num_client_models):
        #         if i < j:
        #             continue
        #         print('FedAvg: cosine distance between {} and {} is {:.8f}'.format(i, j, cd[i][j]))

        return weight_accumulator

    def accumulate_weights(self, weight_accumulator, local_update, scale, reverse_scale=False):
        if reverse_scale is False:
            scale = 1. / scale

        for name, value in local_update.items():
            if name not in weight_accumulator:
                weight_accumulator[name] = value.to(self.device) * scale
                continue
            weight_accumulator[name].add_(value.to(self.device) * scale)

    @staticmethod
    def get_update_norm(local_update):
        squared_sum = 0
        for name, value in local_update.items():
            if 'tracked' in name or 'running' in name:
                continue
            squared_sum += torch.sum(torch.pow(value, 2)).item()
        update_norm = math.sqrt(squared_sum)
        return update_norm

    @staticmethod
    def get_vectorized_params(local_model_state_dict):
        vectorized_params = torch.cat([param.view(-1) for param in local_model_state_dict.values()])
        return vectorized_params

    def add_noise(self, sum_update_tensor: torch.Tensor, sigma):
        noised_layer = torch.FloatTensor(sum_update_tensor.shape)
        noised_layer = noised_layer.to(self.device)
        noised_layer.normal_(mean=0, std=sigma)
        sum_update_tensor.add_(noised_layer)

    def check_ignored_weights(self, name) -> bool:
        for ignored in self.ignored_weights:
            if ignored in name:
                return True

        return False
