import logging
import numpy as np
import networkx as nx

import torch
import torch_geometric.transforms as T
from torch_geometric.utils import scatter
from torch_geometric.utils import to_networkx, from_networkx, remove_self_loops
from torch_geometric.datasets import Planetoid, Reddit2, Flickr, Reddit, Yelp, Coauthor, Amazon, DGraphFin, AttributedGraphDataset

from dataset.amazon import amazon_data
from dataset.coauther import coauthor_data
from dataset.ogba import ogba_data

from dataset.partition import IIDPartitioner, DirichletPartitioner, \
    NodeLouvainPartitioner, NodeMetisPartitioner, NodeOverlappingPartitioner, \
    OneClassPartitioner, TwoClassPartitioner, InterClientNonIIDPartitioner

try:
    if 'logger' not in globals():
        logging.basicConfig()
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
except NameError:
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)


def load_dataset(dataset_name):
    coauthor_list = ['cs', 'physics']
    amazon_list = ['computers', 'photo']
    dataset_name = dataset_name.lower()

    dataset = None
    if dataset_name in ['cora', 'citeseer', 'pubmed']:
        dataset = Planetoid(root='./data', name=dataset_name, transform=T.Compose([
            T.LargestConnectedComponents()
        ]))
    elif dataset_name == 'flickr':
        dataset = Flickr(root='./data/{}/'.format(dataset_name), transform=T.Compose([
            T.LargestConnectedComponents()
        ]))
    elif dataset_name == 'reddit':
        dataset = Reddit(root='./data/reddit/', transform=T.Compose([
            T.LargestConnectedComponents()
        ]))
    elif dataset_name == 'reddit2':
        dataset = Reddit2(root='./data/reddit2/', transform=T.Compose([
            T.LargestConnectedComponents()
        ]))
    elif dataset_name == 'yelp':
        dataset = Yelp(root='./data/yelp/', transform=T.Compose([
            T.LargestConnectedComponents()
        ]))
        # Convert one-hot encoded labels to integer labels
        labels = np.argmax(dataset.data.y.numpy(), axis=1)

        # Create new data object with integer labels
        data = dataset.data
        data.y = torch.from_numpy(labels).reshape(-1).long()
    elif dataset_name == 'graphfin':
        dataset = DGraphFin(root='./data/dgraphfin/', transform=T.Compose([
            T.LargestConnectedComponents()
        ]))
    elif dataset_name == 'tweibo':
        dataset = AttributedGraphDataset(root='./data/TWeibo/', name='TWeibo', transform=T.Compose([
            T.LargestConnectedComponents()
        ]))
    elif dataset_name == 'mag':
        dataset = AttributedGraphDataset(root='./data/MAG/', name='MAG', transform=T.Compose([
            T.LargestConnectedComponents()
        ]))
    elif dataset_name in ['ogbn-arxiv', 'ogbn-products', 'ogbn-proteins', 'ogbn-papers100m']:
        from ogb.nodeproppred import PygNodePropPredDataset
        if dataset_name == 'ogbn-papers100m':
            _dataset_name = 'ogbn-papers100M'
        else:
            _dataset_name = dataset_name
        dataset = PygNodePropPredDataset(name=_dataset_name, root='./data/')
    elif dataset_name in coauthor_list:
        dataset = Coauthor(root='./data/', name=dataset_name, transform=T.Compose([
            T.LargestConnectedComponents()
        ]))
    elif dataset_name in amazon_list:
        dataset = Amazon(root='./data/', name=dataset_name, transform=T.Compose([
            T.LargestConnectedComponents()
        ]))

    # Dataset Statistical Information
    logger.info(f'Dataset: {dataset_name}:')
    logger.info(f'Number of graphs: {len(dataset)}')
    logger.info(f'Number of features: {dataset.num_features}')
    logger.info(f'Number of classes: {dataset.num_classes}')

    ogbn_data_list = ['ogbn-arxiv', 'ogbn-products', 'ogbn-proteins', 'ogbn-papers100m']
    if dataset_name in ogbn_data_list:
        data = ogba_data(dataset)
        data.y = data.y.reshape(-1)
    elif dataset_name in amazon_list:
        data = amazon_data(dataset)
        data.y = data.y.to(dtype=torch.long)
    elif dataset_name in coauthor_list:
        data = coauthor_data(dataset)
    elif dataset_name in ['tweibo']:
        data = amazon_data(dataset)
    else:
        data = dataset[0]  # Get the graph object.

    if dataset_name == 'ogbn-proteins':
        # Initialize features of nodes by aggregating edge features.
        row, col = data.edge_index
        data.x = scatter(data.edge_attr, col, dim_size=data.num_nodes, reduce='sum')
        _, f_dim = data.x.size()
        print(f'ogbn-proteins Number of features: {f_dim}')
        print('data.y = data.y.to(torch.float)', data.y.shape)

    if dataset_name == 'reddit':
        data.y = data.y.long()
    elif dataset_name == 'graphfin':
        data.y = torch.where(torch.bitwise_or(data.y == 2, data.y == 3), torch.tensor(0), data.y)

    # if dataset_name in ['cora']:
    #     data = T.NormalizeFeatures()(data)

    data.edge_index, data.edge_weight = remove_self_loops(data.edge_index, data.edge_weight)
    avg_degree = data.num_edges / data.num_nodes
    num_classes = int(data.y.max() + 1)
    return data, avg_degree, num_classes


def rtn_node_labels(d):
    num_nodes = d.number_of_nodes()
    node_attrs = nx.get_node_attributes(d, 'y')
    node_labels = [node_attrs[node_id] for node_id in range(num_nodes)]
    return node_labels


def dataset_distribution(dataset_name, data, num_clients, distribution_mode, non_iid_degree, target_class):

    if distribution_mode == 'uniform':
        partitioner = IIDPartitioner(num_clients=num_clients)
    elif distribution_mode == 'dirichlet':
        partitioner = DirichletPartitioner(num_clients=num_clients, alpha=non_iid_degree, index_func=lambda g: rtn_node_labels(g))
    elif distribution_mode == 'metis':
        partitioner = NodeMetisPartitioner(num_clients=num_clients)
    elif distribution_mode == 'louvain':
        partitioner = NodeLouvainPartitioner(num_clients=num_clients)
    elif distribution_mode == 'overlapping':
        partitioner = NodeOverlappingPartitioner(num_clients=num_clients)
    elif distribution_mode == 'oneclass':
        partitioner = OneClassPartitioner(num_clients=num_clients - 1, index_func=lambda g: rtn_node_labels(g), target_class=target_class)
    elif distribution_mode == 'twoclass':
        partitioner = TwoClassPartitioner(num_clients=num_clients - 1, index_func=lambda g: rtn_node_labels(g))
    elif distribution_mode == 'inter-client':
        partitioner = InterClientNonIIDPartitioner(num_clients=num_clients - 1, index_func=lambda g: rtn_node_labels(g))
    else:
        raise NotImplementedError('The partitioner [] is not implemented'.format(distribution_mode))

    if dataset_name in ['mag', 'tweibo']:
        nx_graph = to_networkx(data, node_attrs=['x', 'y', 'train_mask', 'val_mask', 'test_mask'], to_undirected=False)
    else:
        nx_graph = to_networkx(data, node_attrs=['x', 'y', 'train_mask', 'val_mask', 'test_mask'], to_undirected=True)
    test_graph = None
    if distribution_mode in ['oneclass', 'twoclass', 'inter-client']:
        node_labels = rtn_node_labels(nx_graph)
        num_nodes = len(node_labels) // num_clients
        test_node_indices = np.random.choice(len(node_labels), num_nodes, replace=False)
        test_graph = nx.subgraph(nx_graph, test_node_indices)
        nx_graph.remove_nodes_from(test_node_indices)

        node_mapping = {old_node: new_node for new_node, old_node in enumerate(nx_graph.nodes(), start=0)}
        nx_graph = nx.relabel_nodes(nx_graph, node_mapping)

    local_node_indices = partitioner(nx_graph)

    graphs = []
    for nodes in local_node_indices:
        sub_g = nx.Graph(nx.subgraph(nx_graph, nodes))
        graphs.append(from_networkx(sub_g))

    if distribution_mode in ['oneclass', 'twoclass', 'inter-client']:
        graphs.append(test_graph)
    return graphs
