import argparse
import wandb

import torch

from utils import seed_experiment
from servers.server import FLServer
from dataset.utils import load_dataset


def calc_adjusted_homophily(edge_index, labels):

    import torch_scatter
    from torch_geometric.utils import degree, to_undirected
    from torch_geometric.utils import homophily

    edge_index = to_undirected(edge_index)

    num_labels = torch.max(labels).item() + 1
    label_degree_cnt = torch.zeros(num_labels, dtype=torch.float)

    node_degrees = degree(edge_index[0])
    label_degree_cnt = torch_scatter.scatter(node_degrees, labels, reduce='sum')

    num_edges = edge_index.shape[1]
    total = torch.sum(label_degree_cnt ** 2) / num_edges ** 2

    edge_hm = homophily(edge_index, labels, method='edge')
    adjusted_homophily = (edge_hm - total) / (1.0 - total)
    return adjusted_homophily


def bool_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def args_parser():
    parser = argparse.ArgumentParser(description='Federated Graph Backdoor Attack')

    # Seeding
    parser.add_argument('--seed', type=int, default=1314, help='seed')

    # Federated Learning Settings
    parser.add_argument('--num_clients', type=int, default=5, help='number of clients')
    parser.add_argument('--num_chosen_clients', type=int, default=5,
                        help='num of clients randomly chosen to participate in federated learning')
    parser.add_argument('--num_malicious', type=int, default=1,
                        help='number of malicious clients')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate for training')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay (L2 loss on parameters)')
    parser.add_argument('--num_rounds', type=int, default=200, help='fl rounds')
    parser.add_argument('--load_model', type=bool_string, default=False, help='path to load the model')
    parser.add_argument('--save_model', type=bool_string, default=False, help='path to save the model')

    # Backdoor Attacks
    parser.add_argument('--attack_name', type=str, default='pgd',
                        choices=['pgd', 'pgd-replace', 'data-poison', 'constrain-scale', 'beef', 'auto-adapt'])
    parser.add_argument('--target_class', type=int, default=0)
    parser.add_argument('--num_start_round',
                        type=int, default=10, help='from which epoch the malicious clients start backdoor attack')
    parser.add_argument('--num_end_round',
                        type=int, default=10000, help='from which epoch the malicious clients start backdoor attack')
    parser.add_argument('--trigger_size', type=int, default=3, choices=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 50, 100], help='trigger size')
    parser.add_argument('--poisoning_intensity', type=float, default=0.1, help='ratio of poisoning nodes relative to the full graph')
    parser.add_argument('--density', type=float, default=0.8, help='density of the edge in the generated trigger')
    parser.add_argument('--degree', type=int, default=3, help='The degree of trigger type')
    parser.add_argument('--trigger_position', type=str, default='random',
                        choices=['learn_cluster', 'random', 'learn_cluster_degree', 'degree', 'cluster'],
                        help='method to select idx_attach for training trojan model')
    parser.add_argument('--trigger_type', type=str, default='renyi', choices=['renyi', 'ws', 'ba', 'gta', 'ugba', 'beef'],
                        help='generate the trigger methods')
    parser.add_argument('--dis_weight', type=float, default=1, help='weight of cluster distance')
    parser.add_argument('--train_epochs', type=int, default=200,
                        help='Number of epochs to train benign and backdoor model in trigger position selection.')
    parser.add_argument('--outer_epochs', type=int, default=400, help='Number of epochs to train trigger generator.')
    parser.add_argument('--inner_epochs', type=int, default=1, help='epochs for training')

    parser.add_argument('--thrd', type=float, default=0.5, help='generation trigger edges')
    parser.add_argument('--target_loss_weight', type=float, default=1, help='weight of optimize outer trigger generator in ugba')
    parser.add_argument('--homo_loss_weight', type=float, default=100, help='weight of optimize similarity loss in ugba')
    parser.add_argument('--dd_loss_weight', type=float, default=100, help='weight of optimize similarity loss in Ada attack')
    parser.add_argument('--homo_boost_thrd', type=float, default=0.8, help='threshold of increase similarity in ugba')

    parser.add_argument('--eps', type=float, default=2.0, help='threshold of pgd attack')
    parser.add_argument('--scale', type=float, default=1.0, help='scale of model weights of malicious clients')
    parser.add_argument('--alpha', type=float, default=10.0, help='hyper-parameters of AutoAdapt')

    parser.add_argument('--tau', type=float, default=0.05, help='threshold for gumbel-softmax')
    parser.add_argument('--n_round', type=int, default=10, help='average number of rounds')
    parser.add_argument('--num_hidden_generator', type=int, default=128, help='hidden dimensions of generators')
    parser.add_argument('--magnitude_thresh_ratio', type=float, default=0.05, help='threshold of perturbed weights')
    parser.add_argument('--balance_factor', type=float, default=1.0, help='balancing attack performance and consistency')
    parser.add_argument('--generator_epochs', type=int, default=5, help='epochs to train generator')

    # Data Settings
    parser.add_argument('--dataset', type=str, default='reddit', help='Dataset',
                        choices=['cora', 'citeseer', 'pubmed', 'flickr', 'ogbn-arxiv', 'reddit', 'reddit2', 'mag', 'tweibo',
                                 'yelp', 'cs', 'physics', 'computers', 'photo', 'graphfin', 'ogbn-products', 'ogbn-proteins', 'ogbn-papers100M'])
    parser.add_argument('--data_distribution', type=str, default='uniform',
                        choices=['uniform', 'dirichlet', 'metis', 'louvain', 'overlapping', 'oneclass', 'twoclass', 'inter-client'],
                        help='split the graph into the clients: random is randomly split, louvain is the community detection method')
    parser.add_argument('--ratio_train', type=float, default=0.4, help='labels of ratio of training')
    parser.add_argument('--ratio_val', type=float, default=0.1, help='labels of ratio of val')
    parser.add_argument('--ratio_test', type=float, default=0.2, help='labels of ratio of testing')
    parser.add_argument('--non_iid_degree', type=float, default=0.0, choices=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                        help='alpha parameter to control dirichlet distribution')

    # Model Settings
    parser.add_argument('--model', type=str, default='GraphSage', help='model', choices=['GCN', 'GAT', 'GraphSage', 'SGC', 'APPNP', 'GNNGuard', 'RobustGCN', 'MedianGCN', 'ElasticGNN'])
    parser.add_argument('--num_hidden', type=int, default=32, help='size of GNN hidden layer')
    parser.add_argument('--local_epochs', type=int, default=1, help='epochs for training')
    parser.add_argument('--adv_local_epochs', type=int, default=0, help='epochs for training')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate (1 - keep probability).')

    # Defense Settings
    parser.add_argument('--defense_name', type=str, default='none',
                        choices=['flame', 'foolsgold', 'krum', 'fedavg', 'fedprox', 'fedpub', 'auror', 'robustlr', 'deepsight', 'mesas', 'freqfed'])
    parser.add_argument('--lamda', type=float, default=0.001, help='magnitude of noise')
    parser.add_argument('--mu', type=float, default=0.01, help='balance factor of regularize')
    parser.add_argument('--l1', type=float, default=1e-3, help='the threshold for masking on FedPub')
    parser.add_argument('--loc_l2', type=float, default=1e-3, help='the threshold for masking n FedPub')
    parser.add_argument('--aggr_norm', type=str, default='exp', help='preprocess operator for similarity matrix')
    parser.add_argument('--norm_scale', type=float, default=10, help='preprocess operator parameter')
    parser.add_argument('--auror_alpha', type=float, default=10, help='Auror Parameter')
    parser.add_argument('--auror_tau', type=float, default=10, help='Auror Parameter')
    parser.add_argument('--rlr_theta', type=float, default=8, help='hyperprameters of Robustrlr')
    parser.add_argument('--rlr_lr', type=float, default=0.01, help='preprocess operator parameter')

    # Misc
    parser.add_argument('--device_id', type=int, default=0, help='device id')
    parser.add_argument('--proj_name', type=str, default='Federated Graph Backdoor Attacks', help='wandb logger project name')
    parser.add_argument('--group_name', type=str, default='test', help='wandb logger group name')
    return parser.parse_args()


if __name__ == '__main__':
    args = args_parser()
    seed_experiment(args.seed)

    logger = wandb.init(
        project=args.proj_name,
        group=args.group_name,
        config=vars(args),
    )
    # logger = None

    if args.device_id >= 0:
        device = f'cuda:{args.device_id}'
    else:
        device = 'cpu'

    # Loading Datasets
    data, avg_degree, num_classes = load_dataset(args.dataset)

    # Calculate homogeneity degree
    # homophily = calc_adjusted_homophily(data.edge_index, data.y)
    # print('{} Dataset homophily = {:.4f}'.format(args.dataset, homophily.item()))
    # exit(0)

    model_state_dict = None
    if args.load_model:
        model_path = 'checkpoints/{}_{}_80_{}.pt'.format(args.model, args.dataset, args.data_distribution)
        model_state_dict = torch.load(model_path)

    if args.adv_local_epochs == 0:
        args.adv_local_epochs = args.local_epochs

    server = FLServer(
        args.num_clients, args.num_malicious, args.num_chosen_clients, data, args.data_distribution,
        args.non_iid_degree, args.dataset, model_state_dict, args.model, args.attack_name, args.defense_name, args.trigger_type,
        args.trigger_position, args.poisoning_intensity, args.target_class, args.trigger_size, args.thrd, args.ratio_train, args.ratio_val,
        args.ratio_test, args.target_loss_weight, args.homo_loss_weight, args.dd_loss_weight, args.homo_boost_thrd, args.density, args.degree,
        args.dis_weight, args.train_epochs, args.outer_epochs, args.inner_epochs, args.lr, args.weight_decay, args.local_epochs, args.adv_local_epochs, args.num_hidden,
        num_classes, args.dropout, args.eps, args.scale, args.seed, device, args.num_rounds, args.mu, args.num_start_round, args.num_end_round, args.num_hidden_generator,
        args.magnitude_thresh_ratio, args.tau, args.alpha, args.n_round, args.balance_factor, args.generator_epochs, args.lamda, args.l1, args.loc_l2,
        args.aggr_norm, args.norm_scale, args.auror_alpha, args.auror_tau, args.rlr_theta, args.rlr_lr, logger
    )
    model = server.exec()

    if args.save_model:
        model_state_dict = model.state_dict()
        torch.save(model_state_dict, 'checkpoints/{}_{}_{}_{}.pt'.format(args.model, args.dataset, args.num_rounds, args.data_distribution))
