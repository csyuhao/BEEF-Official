import random
import argparse
from collections import Counter

import numpy as np
from copy import deepcopy

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
from torch_geometric.loader import DataLoader

from data import load_dataset
from model import GCN
from beef.generator import GenModel
from utils import separate_data, average_weights, random_subgraphs


def main():

    parser = argparse.ArgumentParser(description='PyTorch graph convolutional neural net for whole-graph classification')
    parser.add_argument('--dataset', type=str, default="MUTAG", help='name of dataset (default: MUTAG)')
    parser.add_argument('--num_agents', type=int, default=10, help='number of agents (default: 10)')
    parser.add_argument('--target_label', type=int, default=0, help='target label (default: 0)')
    parser.add_argument('--num_corrupt', type=int, default=2, help='number of corruptions (default: 0)')
    parser.add_argument('--trigger_size', type=int, default=3, help='size of trigger (default: 3)')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size (default: 64)')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs (default: 200)')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.001)')
    parser.add_argument('--attack_method', type=str, default='BEEF', help='attack method (default: BEEF)')
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dataset, num_classes = load_dataset(args.dataset)
    train_dataset, test_dataset = separate_data(dataset)
    num_feat = dataset._data.num_features

    train_dataset_size = len(train_dataset)
    client_data_size = int(train_dataset_size / args.num_agents)
    split_data_size = [client_data_size for i in range(args.num_agents - 1)]
    split_data_size.append(train_dataset_size - client_data_size * (args.num_agents - 1))
    train_graphs = data.random_split(train_dataset, split_data_size)

    global_model = GCN(n_feat=num_feat, n_class=num_classes, n_hid=64, dropout=0.5).to(device)

    optimizer = optim.Adam(global_model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    generator = {}
    optimizer_G = {}
    for g_i in range(args.num_corrupt):
        if args.attack_method == 'BEEF':
            generator[g_i] = GenModel(feat_dim=num_feat, hid_dim=64, trigger_size=args.trigger_size).to(device)
        optimizer_G[g_i] = optim.Adam(generator[g_i].parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        local_weights = []
        if (epoch + 1) % 10 == 0:
            print('Conducting epoch = {}'.format(epoch + 1))
        for i in range(args.num_agents):

            if i < args.num_corrupt:
                train_graph = train_graphs[i]
                indices = torch.randperm(len(train_graph)).tolist()
                random.shuffle(indices)

                global_model.eval()
                for j, idx in enumerate(indices):
                    generator[i].train()
                    optimizer_G[i].zero_grad()
                    label = torch.tensor((args.target_label,), dtype=torch.int64, device=device)
                    poisoned_graph = generator[i](train_graph[idx].x.to(device), train_graph[idx].edge_index.to(device))

                    logits = global_model(poisoned_graph.x, poisoned_graph.edge_index, edge_weight=poisoned_graph.edge_weight)
                    loss = F.cross_entropy(logits, label.to(device))
                    loss.backward()
                    optimizer_G[i].step()

            global_model.train()
            train_loader = DataLoader(train_graphs[i], batch_size=args.batch_size, shuffle=True)
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                out = global_model(batch.x, batch.edge_index, batch=batch.batch)
                loss = F.cross_entropy(out, batch.y)
                loss.backward()
                optimizer.step()
            scheduler.step()

            local_weights.append(deepcopy(global_model.state_dict()))

        global_weights = average_weights(local_weights)
        global_model.load_state_dict(global_weights)

    clean_acc = 0
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    for batch in test_loader:
        batch = batch.to(device)
        global_model.eval()
        with torch.no_grad():
            out = global_model(batch.x, batch.edge_index, batch=batch.batch)
            pred = out.argmax(dim=1)
            clean_acc += (pred == batch.y).sum().item()

    backdoor_acc_list = []
    indices = torch.randperm(len(test_dataset)).tolist()
    random.shuffle(indices)

    global_model.eval()
    for i in range(args.num_corrupt):
        backdoor_acc = 0
        for idx in indices:
            generator[i].eval()
            graph = test_dataset[idx].to(device)
            poisoned_graph = generator[i](graph.x, graph.edge_index)
            logits = global_model(poisoned_graph.x, poisoned_graph.edge_index, poisoned_graph.edge_weight)
            backdoor_acc += (logits.argmax(dim=1) == args.target_label).sum().item()
        backdoor_acc_list.append(backdoor_acc)
    backdoor_acc = np.mean(backdoor_acc_list)

    print('Clean Accuracy: {:.4f}@@@Backdoor Accuracy: {:.4f}'.format(clean_acc / len(test_dataset), backdoor_acc / len(test_dataset)))

    # ====================================== Defense ========================================
    indices = torch.randperm(len(test_dataset)).tolist()
    defense_acc = 0
    global_model.eval()

    for idx in indices:
        graph = test_dataset[idx].to(device)

        subgraph_list = random_subgraphs(graph, 10)

        pred_list = []
        for subgraph in subgraph_list:
            subgraph = subgraph.to(device)
            logits = global_model(subgraph.x, subgraph.edge_index)
            pred_list.append(logits.argmax(dim=1).item())
        pred = Counter(pred_list).most_common(1)[0][0]

        defense_acc += pred == test_dataset[idx].y.item()

    global_model.eval()
    backdoor_acc_list = []
    indices = torch.randperm(len(test_dataset)).tolist()
    random.shuffle(indices)

    for i in range(args.num_corrupt):
        backdoor_acc = 0
        for idx in indices:
            generator[i].eval()
            graph = test_dataset[idx].to(device)
            poisoned_graph = generator[i](graph.x, graph.edge_index)
            subgraph_list = random_subgraphs(poisoned_graph, 10)

            pred_list = []
            for subgraph in subgraph_list:
                subgraph = subgraph.to(device)
                logits = global_model(subgraph.x, subgraph.edge_index, subgraph.edge_weight)
                pred_list.append(logits.argmax(dim=1).item())
            pred = Counter(pred_list).most_common(1)[0][0]
            backdoor_acc += (pred == args.target_label)
        backdoor_acc_list.append(backdoor_acc)
    backdoor_defense_acc = np.mean(backdoor_acc_list)

    print('Defense Accuracy: {:.4f}@Defense Backdoor Accuracy: {:.4f}'.format(
        defense_acc / len(test_dataset), backdoor_defense_acc / len(test_dataset)
    ))


if __name__ == '__main__':
    main()
