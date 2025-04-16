import numpy as np

import torch


def local_data_split(data, ratio_train, ratio_val, ratio_test, seed=1314):
    rs = np.random.RandomState(seed)
    perm = rs.permutation(data.num_nodes)
    train_num_nodes = int(ratio_train * len(perm))

    idx_train = torch.tensor(sorted(perm[:train_num_nodes]))
    data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.train_mask[idx_train] = True

    val_num_nodes = int(ratio_val * len(perm))
    idx_val = torch.tensor(sorted(perm[train_num_nodes: train_num_nodes + val_num_nodes]))
    data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.val_mask[idx_val] = True

    test_num_nodes = int(ratio_test * len(perm))
    idx_test = torch.tensor(sorted(perm[train_num_nodes + val_num_nodes: train_num_nodes + val_num_nodes + test_num_nodes]))
    data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.test_mask[idx_test] = True

    idx_clean_test = idx_test[:int(len(idx_test) / 2)]
    idx_attack = idx_test[int(len(idx_test) / 2):]
    return data, idx_train, idx_val, idx_clean_test, idx_attack


def subgraph(subset, edge_index, edge_attr=None):
    node_mask = subset
    edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
    edge_index = edge_index[:, edge_mask]
    edge_attr = edge_attr[edge_mask] if edge_attr is not None else None
    return edge_index, edge_attr, edge_mask
