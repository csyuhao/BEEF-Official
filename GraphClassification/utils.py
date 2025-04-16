import copy
import hashlib
import numpy as np
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Subset
from torch_geometric.data import Data


def separate_data(data):
    indices = np.arange(len(data))
    train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42)

    train_dataset = Subset(data, train_indices)
    test_dataset = Subset(data, test_indices)
    return train_dataset, test_dataset


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def random_subgraphs(data, num_subgraphs):
    """
    将 Data 对象随机切分为多个子图

    Args:
        data (Data): 父图的 Data 对象
        num_subgraphs (int): 子图数量

    Returns:
        list[Data]: 子图列表
    """
    node_ids = torch.arange(data.num_nodes)
    subgraph_node_list = [[] for _ in range(num_subgraphs)]
    for node_id in node_ids:
        hashed_val = int(hashlib.md5(str(node_id).encode()).hexdigest(), 16) % num_subgraphs
        sub_idx = hashed_val
        subgraph_node_list[sub_idx].append(node_id.item())

    edge_list = data.edge_index.cpu().numpy().T
    subgraph_edge_list = []
    for i, j in edge_list:
        i, j = (i, j) if i < j else (j, i)
        txt = '{}{}'.format(i, j)
        hash_val = int(hashlib.md5(txt.encode()).hexdigest(), 16) % num_subgraphs
        subgraph_edge_list.append(hash_val)
    subgraph_edge_list = np.array(subgraph_edge_list, dtype=np.int64)

    subgraphs = []
    for i in range(num_subgraphs):
        selected_nodes = subgraph_node_list[i]
        if not selected_nodes:
            continue  # 跳过空子图

        edge_mask = (subgraph_edge_list == i)

        # 继承节点特征和标签
        if hasattr(data, 'edge_weight') and data.edge_weight is not None:
            sub_data = Data(x=data.x, edge_index=data.edge_index[:, edge_mask], y=data.y,
                            edge_weight=data.edge_weight[edge_mask], num_nodes=data.x.shape[0], device=data.x.device)
        else:
            sub_data = Data(x=data.x, edge_index=data.edge_index[:, edge_mask],
                            y=data.y, num_nodes=data.x.shape[0], device=data.x.device)

        subgraphs.append(sub_data)
    return subgraphs
