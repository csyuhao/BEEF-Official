import os
import time
import numpy as np
from sklearn_extra import cluster
from sklearn.cluster import KMeans

import torch

import dgl
import networkx as nx

from models.utils import model_construct
from utils import fit_model, eval_model


def max_norm(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def obtain_attach_nodes(node_idxs, size, seed=1314):
    size = min(len(node_idxs), size)
    rs = np.random.RandomState(seed)
    choice = np.arange(len(node_idxs))
    rs.shuffle(choice)
    return node_idxs[choice[:size]]


def cluster_distance_selection(data, idx_train, idx_val, idx_clean_test, idx_unlabelled,
                               train_edge_index, size, dataset_name, num_hidden, num_classes,
                               dropout, lr, weight_decay, train_epochs, dis_weight, target_class, device):
    encoder_model_path = './checkpoints/{}_{}_benign.pth'.format('GCN_Encoder', dataset_name)

    if os.path.exists(encoder_model_path):
        # load existing benign model
        gcn_encoder = torch.load(encoder_model_path)
        gcn_encoder = gcn_encoder.to(device)
        print('Loading {} encoder Finished!'.format('GCN_Encoder'))
    else:
        gcn_encoder = model_construct(data, num_hidden, num_classes, dropout, 'GCN_Encoder').to(device)
        begin_time = time.time()
        print('Length of training set: {}'.format(len(idx_train)))

        gcn_encoder = fit_model(gcn_encoder, lr, weight_decay, train_epochs, data, train_edge_index, idx_train, idx_val, device)

        print('Training encoder Finished!')
        print('Total time elapsed: {:.4f}s'.format(time.time() - begin_time))

    # test gcn encoder
    encoder_clean_test_ca = eval_model(gcn_encoder, data, idx_clean_test, device)
    print('Encoder CA on clean test nodes: {:.4f}'.format(encoder_clean_test_ca))

    gcn_encoder.eval()
    seen_node_idx = torch.concat([idx_train.to(device), idx_unlabelled.to(device)])
    num_classes = np.unique(data.y.cpu().numpy()).shape[0]
    encoder_x = gcn_encoder.get_h(data.x.to(device), train_edge_index.to(device), None).clone().detach()
    encoder_output = gcn_encoder(data.x.to(device), train_edge_index.to(device), None)

    y_pred = np.array(encoder_output.argmax(dim=1).cpu()).astype(int)
    kmedoids = cluster.KMedoids(n_clusters=num_classes, method='pam')
    kmedoids.fit(encoder_x[seen_node_idx].detach().cpu().numpy())
    idx_attach = obtain_attach_nodes_by_cluster(
        kmedoids, y_pred, idx_unlabelled.cpu().tolist(), encoder_x, dis_weight, target_class, size
    ).astype(int)

    return idx_attach


def obtain_attach_nodes_by_cluster(model, y_pred, node_idxs, x, dis_weight, target_class, size):
    cluster_centers = model.cluster_centers_

    distances = []
    distances_tar = []
    for idx in range(x.shape[0]):
        tmp_center_label = y_pred[idx]
        tmp_tar_label = target_class

        tmp_center_x = cluster_centers[tmp_center_label]
        tmp_tar_x = cluster_centers[tmp_tar_label]

        dis = np.linalg.norm(tmp_center_x - x[idx].detach().cpu().numpy())
        dis_tar = np.linalg.norm(tmp_tar_x - x[idx].cpu().numpy())
        distances.append(dis)
        distances_tar.append(dis_tar)

    distances = np.array(distances)
    distances_tar = np.array(distances_tar)
    label_list = np.unique(y_pred)
    labels_dict = {}
    for i in label_list:
        labels_dict[i] = np.where(y_pred == i)[0]
        labels_dict[i] = np.array(list(set(node_idxs) & set(labels_dict[i])))

    each_selected_num = int(size / len(label_list) - 1)
    last_selected_num = size - each_selected_num * (len(label_list) - 2)
    candidate_nodes = np.array([])
    for label in label_list:

        if label == target_class:
            continue

        single_labels_nodes = labels_dict[label]  # the node idx of the nodes in single class
        single_labels_nodes = np.array(list(set(single_labels_nodes)))
        print('single_labels_nodes = [{}]'.format(', '.join(['{}'.format(n) for n in single_labels_nodes])))
        single_labels_nodes_dis = distances[single_labels_nodes]
        single_labels_nodes_dis = max_norm(single_labels_nodes_dis)
        single_labels_nodes_dis_tar = distances_tar[single_labels_nodes]
        single_labels_nodes_dis_tar = max_norm(single_labels_nodes_dis_tar)

        # the closer to the center, the more far away from the target centers
        single_labels_dis_score = dis_weight * single_labels_nodes_dis + (-single_labels_nodes_dis_tar)
        single_labels_nid_index = np.argsort(single_labels_dis_score)  # sort decently based on the distance away from the center
        sorted_single_labels_nodes = np.array(single_labels_nodes[single_labels_nid_index])

        if label != label_list[-1]:
            candidate_nodes = np.concatenate([candidate_nodes, sorted_single_labels_nodes[:each_selected_num]])
        else:
            candidate_nodes = np.concatenate([candidate_nodes, sorted_single_labels_nodes[:last_selected_num]])

    return candidate_nodes


def cluster_degree_selection(data, idx_train, idx_val, idx_clean_test, idx_unlabelled,
                             train_edge_index, size, dataset_name, num_hidden, num_classes,
                             dropout, lr, weight_decay, train_epochs, dis_weight, target_class, seed, device):

    gcn_encoder = model_construct(data, num_hidden, num_classes, dropout, 'GCN_Encoder').to(device)

    if os.path.exists('./checkpoints/aux_models/{}_{}_{}.pth'.format('GCN_Encoder', dataset_name, seed)):
        # load existing benign model
        state_dict = torch.load(
            './checkpoints/aux_models/{}_{}_{}.pth'.format('GCN_Encoder', dataset_name, seed),
            map_location=device
        )
        gcn_encoder.load_state_dict(state_dict)
        gcn_encoder = gcn_encoder.to(device)
        print('Loading {} encoder Finished!'.format('GCN_Encoder'))
    else:
        begin_time = time.time()
        print('Length of training set: {}'.format(len(idx_train)))
        gcn_encoder = fit_model(gcn_encoder, lr, weight_decay, train_epochs, data, train_edge_index, idx_train, idx_val, device)
        print('Training encoder Finished!')
        print('Total time elapsed: {:.4f}s'.format(time.time() - begin_time))
        torch.save(gcn_encoder.state_dict(), './checkpoints/aux_models/{}_{}_{}.pth'.format('GCN_Encoder', dataset_name, seed))

    encoder_clean_test_ca = eval_model(gcn_encoder, data, idx_clean_test, device)
    print('Encoder CA on clean test nodes: {:.4f}'.format(encoder_clean_test_ca))

    idx_train, idx_unlabelled = idx_train.to(device), idx_unlabelled.to(device)
    seen_node_idx = torch.concat([idx_train, idx_unlabelled])
    num_classes = np.unique(data.y.cpu().numpy()).shape[0]
    encoder_x = gcn_encoder.get_h(data.x.to(device), train_edge_index.to(device), None).clone().detach()

    if dataset_name == 'cora' or dataset_name == 'citeseer':
        kmedoids = cluster.KMedoids(n_clusters=num_classes, method='pam', random_state=1314)
        kmedoids.fit(encoder_x[seen_node_idx].detach().cpu().numpy())
        cluster_centers = kmedoids.cluster_centers_
        y_pred = kmedoids.predict(encoder_x.cpu().numpy())
    else:
        kmeans = KMeans(n_clusters=num_classes, random_state=1314, n_init='auto')
        kmeans.fit(encoder_x[seen_node_idx].detach().cpu().numpy())
        cluster_centers = kmeans.cluster_centers_
        y_pred = kmeans.predict(encoder_x.cpu().numpy())

    idx_attach = obtain_attach_nodes_by_cluster_degree_all(
        train_edge_index, y_pred, idx_unlabelled.cpu().tolist(), encoder_x, dis_weight, cluster_centers, target_class
    ).astype(int)
    idx_attach = idx_attach[:size]
    return idx_attach


def obtain_attach_nodes_by_cluster_degree_all(edge_index, y_pred, node_idxs, x, dis_weight, cluster_centers, target_class):

    device = edge_index.device
    deg = torch.zeros(len(y_pred), dtype=torch.long).to(device)
    deg.index_add_(0, edge_index[0], torch.ones_like(edge_index[1], device=device))
    deg.index_add_(0, edge_index[1], torch.ones_like(edge_index[0], device=device))
    degrees = deg.cpu().numpy()

    distances = []
    for idx in range(x.shape[0]):
        tmp_center_label = y_pred[idx]
        tmp_center_x = cluster_centers[tmp_center_label]

        dis = np.linalg.norm(tmp_center_x - x[idx].detach().cpu().numpy())
        distances.append(dis)

    distances = np.array(distances)
    non_target_nodes = np.where(y_pred != target_class)[0]

    non_target_node_idxs = np.array(list(set(non_target_nodes) & set(node_idxs)))
    node_idxs = np.array(non_target_node_idxs)
    candidate_distances = distances[node_idxs]
    candidate_degrees = degrees[node_idxs]
    candidate_distances = max_norm(candidate_distances)
    candidate_degrees = max_norm(candidate_degrees)

    dis_score = candidate_distances + dis_weight * candidate_degrees
    candidate_nid_index = np.argsort(dis_score)

    sorted_node_idx = np.array(node_idxs[candidate_nid_index])
    selected_nodes = sorted_node_idx
    return selected_nodes


def sort_subset(sorted_nodes, node_idxs):
    node_idx_dict = {node_idx: i for i, node_idx in enumerate(sorted_nodes)}
    sorted_node_idxs = sorted(node_idxs.numpy(), key=lambda x: node_idx_dict[x])
    return sorted_node_idxs


def obtain_attach_nodes_degree(node_idxs, edge_index, num_nodes, size):

    # Calculate the degree of each node
    deg = torch.zeros(num_nodes, dtype=torch.long, device=edge_index.device)
    deg.index_add_(0, edge_index[0], torch.ones_like(edge_index[1]))
    deg.index_add_(0, edge_index[1], torch.ones_like(edge_index[0]))

    # Create a dictionary where keys are node indices and values are degrees
    degree_dict = dict(enumerate(deg.tolist()))

    # Sort the nodes by their degree in descending order
    sorted_nodes = sorted(degree_dict, key=degree_dict.get, reverse=True)
    sorted_node_idxs = sort_subset(sorted_nodes, node_idxs)

    return sorted_node_idxs[:size]


def obtain_attach_nodes_cluster(node_idxs, edge_index, num_nodes, size):

    # create a DGL graph from edge_index
    g = dgl.graph((edge_index[0], edge_index[1]), num_nodes=num_nodes, device=edge_index.device)

    # convert DGL graph to NetworkX graph
    g = dgl.to_networkx(g.cpu())

    #  sort according to cluster
    simple_g = nx.Graph(g)
    clustering_dict = nx.clustering(simple_g, weight='weight')
    sorted_nodes = sorted(clustering_dict, key=clustering_dict.get, reverse=True)
    sorted_node_idxs = sort_subset(sorted_nodes, node_idxs)

    return sorted_node_idxs[:size]
