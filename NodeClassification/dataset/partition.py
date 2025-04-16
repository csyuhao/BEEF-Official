import warnings
import collections

import random
import networkx as nx
import numpy as np
from abc import abstractmethod, ABCMeta

import metispy as metis
import community.community_louvain


class AbstractPartitioner(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass


class BasicPartitioner(AbstractPartitioner):
    """This is the basic class of data partitioner. The partitioner will be directly called by the
    task generator of different benchmarks. By overwriting __call__ method, different partitioner
    can be realized. The input of __call__ is usually a dataset.
    """

    def __init__(self):
        super(BasicPartitioner, self).__init__()
        self.generator = None

    def __call__(self, *args, **kwargs):
        pass

    def register_generator(self, generator):
        """Register the generator as a self's attribute"""
        self.generator = generator

    def data_imbalance_generator(self, num_clients, datasize, imbalance=0, minvol=1):
        """
        Split the data size into several parts

        Args:
            num_clients (int): the number of clients
            datasize (int): the total data size
            imbalance (float): the degree of data imbalance across clients
            minvol (int): the minimal size of dataset
        Returns:
            a list of integer numbers that represents local data sizes
        """
        if imbalance == 0:
            samples_per_client = [int(datasize / num_clients) for _ in range(num_clients)]
            for _ in range(datasize % num_clients):
                samples_per_client[_] += 1
        else:
            imbalance = max(0.1, imbalance)
            sigma = imbalance
            mean_datasize = datasize / num_clients
            mu = np.log(mean_datasize) - sigma ** 2 / 2.0
            samples_per_client = np.random.lognormal(mu, sigma, (num_clients)).astype(int)
            crt_data_size = sum(samples_per_client)
            total_delta = np.abs(crt_data_size - datasize)
            threshold = max(int(total_delta / 10), 1)
            delta = min(int(0.1 * threshold), 10)
            # force current data size to match the total data size
            while crt_data_size != datasize:
                if crt_data_size - datasize >= threshold:
                    maxid = np.argmax(samples_per_client)
                    maxvol = samples_per_client[maxid]
                    new_samples = np.random.lognormal(mu, sigma, (10 * num_clients))
                    while min(new_samples) > maxvol:
                        new_samples = np.random.lognormal(mu, sigma, (10 * num_clients))
                    new_size_id = np.argmin(
                        [np.abs(crt_data_size - samples_per_client[maxid] + s - datasize) for s in new_samples])
                    samples_per_client[maxid] = new_samples[new_size_id]
                elif crt_data_size - datasize >= delta:
                    maxid = np.argmax(samples_per_client)
                    if samples_per_client[maxid] >= delta:
                        samples_per_client[maxid] -= delta
                    elif samples_per_client[maxid] > 1:
                        samples_per_client[maxid] -= 1
                elif crt_data_size - datasize > 0:
                    maxid = np.argmax(samples_per_client)
                    crt_delta = (crt_data_size - datasize)
                    if samples_per_client[maxid] >= crt_delta:
                        samples_per_client[maxid] -= crt_delta
                    elif samples_per_client[maxid] >= minvol:
                        samples_per_client[maxid] -= (crt_delta - minvol)
                    else:
                        warnings.warn("Failed to keep the minvol of clients' training data to be larger than {}".format(minvol))
                        if samples_per_client[maxid] > 1:
                            samples_per_client[maxid] -= 1
                        else:
                            raise RuntimeError(
                                "Failed to generate distribution due to the conflicts of imbalance and num_clients. Please try to decrease the imbalance term or decrease the number of clients. ")
                elif datasize - crt_data_size >= threshold:
                    minid = np.argmin(samples_per_client)
                    minvol = samples_per_client[minid]
                    new_samples = np.random.lognormal(mu, sigma, (10 * num_clients))
                    while max(new_samples) < minvol:
                        new_samples = np.random.lognormal(mu, sigma, (10 * num_clients))
                    new_size_id = np.argmin(
                        [np.abs(crt_data_size - samples_per_client[minid] + s - datasize) for s in new_samples])
                    samples_per_client[minid] = new_samples[new_size_id]
                elif datasize - crt_data_size >= delta:
                    minid = np.argmin(samples_per_client)
                    samples_per_client[minid] += delta
                else:
                    minid = np.argmin(samples_per_client)
                    samples_per_client[minid] += (datasize - crt_data_size)
                crt_data_size = sum(samples_per_client)
            # let the minimal data size to be larger than 0
            while min(samples_per_client) == 0:
                zero_client_idx = np.argmin(samples_per_client)
                maxid = np.argmax(samples_per_client)
                samples_per_client[maxid] -= 1
                samples_per_client[zero_client_idx] += 1
            assert datasize == sum(samples_per_client) and min(samples_per_client) > 0
        return samples_per_client


class IIDPartitioner(BasicPartitioner):
    """`Partition the indices of samples in the original dataset indentically and independently.

    Args:
        num_clients (int, optional): the number of clients
        imbalance (float, optional): the degree of imbalance of the amounts of different local_movielens_recommendation data (0<=imbalance<=1)
    """

    def __init__(self, num_clients=100, imbalance=0):
        super(IIDPartitioner, self).__init__()
        self.num_clients = num_clients
        self.imbalance = imbalance

    def __str__(self):
        name = "iid"
        if self.imbalance > 0:
            name += '_imb{:.1f}'.format(self.imbalance)
        return name

    def __call__(self, data):
        samples_per_client = self.data_imbalance_generator(self.num_clients, len(data), self.imbalance)
        d_idxs = np.random.permutation(len(data))
        local_datas = np.split(d_idxs, np.cumsum(samples_per_client))[:-1]
        local_datas = [di.tolist() for di in local_datas]
        return local_datas


class DirichletPartitioner(BasicPartitioner):
    """`Partition the indices of samples in the original dataset according to Dirichlet distribution of the
    particular attribute. This way of partition is widely used by existing works in federated learning.

    Args:
        num_clients (int, optional): the number of clients
        alpha (float, optional): `alpha`(i.e. alpha>=0) in Dir(alpha*p) where p is the global distribution. The smaller alpha is, the higher heterogeneity the data is.
        imbalance (float, optional): the degree of imbalance of the amounts of different local_movielens_recommendation data (0<=imbalance<=1)
        error_bar (float, optional): the allowed error when the generated distribution mismatches the distirbution that is actually wanted, since there may be no solution for particular imbalance and alpha.
        index_func (func, optional): to index the distribution-dependent (i.e. label) attribute in each sample.
    """

    def __init__(self, num_clients=100, alpha=1.0, error_bar=1e-6, imbalance=0, index_func=lambda X: [xi[-1] for xi in X], minvol=1):
        super(DirichletPartitioner, self).__init__()
        self.num_clients = num_clients
        self.alpha = alpha
        self.imbalance = imbalance
        self.index_func = index_func
        self.minvol = minvol
        self.error_bar = error_bar

    def __str__(self):
        name = "dir{:.2f}_err{}".format(self.alpha, self.error_bar)
        if self.imbalance > 0: name += '_imb{:.1f}'.format(self.imbalance)
        return name

    def __call__(self, data):
        attrs = self.index_func(data)
        num_attrs = len(set(attrs))
        samples_per_client = self.data_imbalance_generator(self.num_clients, len(data), self.imbalance, minvol=self.minvol)
        # count the label distribution
        lb_counter = collections.Counter(attrs)
        lb_names = list(lb_counter.keys())
        p = np.array([1.0 * v / len(data) for v in lb_counter.values()])
        lb_dict = {}
        attrs = np.array(attrs)
        for lb in lb_names:
            lb_dict[lb] = np.where(attrs == lb)[0]
        proportions = [np.random.dirichlet(self.alpha * p) for _ in range(self.num_clients)]
        while np.any(np.isnan(proportions)):
            proportions = [np.random.dirichlet(self.alpha * p) for _ in range(self.num_clients)]
        sorted_cid_map = {k: i for k, i in zip(np.argsort(samples_per_client), [_ for _ in range(self.num_clients)])}
        error_increase_interval = 500
        max_error = self.error_bar
        loop_count = 0
        crt_id = 0
        crt_error = 100000
        while True:
            if loop_count >= error_increase_interval:
                loop_count = 0
                max_error = max_error * 10
            # generate dirichlet distribution till ||E(proportion) - P(D)||<=1e-5*self.num_classes
            mean_prop = np.sum([pi * di for pi, di in zip(proportions, samples_per_client)], axis=0)
            mean_prop = mean_prop / mean_prop.sum()
            error_norm = ((mean_prop - p) ** 2).sum()
            if crt_error - error_norm >= max_error:
                print("Error: {:.8f}".format(error_norm))
                crt_error = error_norm
            if error_norm <= max_error:
                break
            excid = sorted_cid_map[crt_id]
            crt_id = (crt_id + 1) % self.num_clients
            sup_prop = [np.random.dirichlet(self.alpha * p) for _ in range(self.num_clients)]
            del_prop = np.sum([pi * di for pi, di in zip(proportions, samples_per_client)], axis=0)
            del_prop -= samples_per_client[excid] * proportions[excid]
            alter_norms = []
            for i in range(error_increase_interval - loop_count):
                alter_norms = []
                for cid in range(self.num_clients):
                    if np.any(np.isnan(sup_prop[cid])):
                        continue
                    alter_prop = del_prop + samples_per_client[excid] * sup_prop[cid]
                    alter_prop = alter_prop / alter_prop.sum()
                    error_alter = ((alter_prop - p) ** 2).sum()
                    alter_norms.append(error_alter)
                if min(alter_norms) < error_norm:
                    break
            if len(alter_norms) > 0 and min(alter_norms) < error_norm:
                alcid = np.argmin(alter_norms)
                proportions[excid] = sup_prop[alcid]
            loop_count += 1
        local_datas = [[] for _ in range(self.num_clients)]
        self.dirichlet_dist = []  # for efficiently visualizing
        for lb in lb_names:
            lb_idxs = lb_dict[lb]
            lb_proportion = np.array([pi[lb_names.index(lb)] * si for pi, si in zip(proportions, samples_per_client)])
            lb_proportion = lb_proportion / lb_proportion.sum()
            lb_proportion = (np.cumsum(lb_proportion) * len(lb_idxs)).astype(int)[:-1]
            lb_datas = np.split(lb_idxs, lb_proportion)
            self.dirichlet_dist.append([len(lb_data) for lb_data in lb_datas])
            local_datas = [local_data + lb_data.tolist() for local_data, lb_data in zip(local_datas, lb_datas)]
        self.dirichlet_dist = np.array(self.dirichlet_dist).T
        for i in range(self.num_clients):
            np.random.shuffle(local_datas[i])
        len_dist = [len(d) for d in local_datas]
        while min(len_dist) <= self.minvol:
            min_did = np.argmin(len_dist)
            max_did = np.argmax(len_dist)
            max_d = local_datas[max_did]
            min_d = local_datas[min_did]
            if len(max_d) <= self.minvol:
                raise RuntimeError(
                    "The number of clients is too large to distribute enough samples to each client when minvol=={}. Please decrease the number of clients".format(
                        self.minvol)
                )
            min_d.extend(max_d[:1])
            max_d = max_d[1:]
            local_datas[min_did] = min_d
            local_datas[max_did] = max_d
            len_dist = [len(d) for d in local_datas]
        self.local_datas = local_datas
        return local_datas


class NodeLouvainPartitioner(BasicPartitioner):
    """
    Partition a graph into several subgraph by louvain algorithms. The input
    of this partitioner should be of type networkx.Graph
    """

    def __init__(self, num_clients=100):
        super(NodeLouvainPartitioner, self).__init__()
        self.num_clients = num_clients

    def __str__(self):
        name = "Louvain"
        return name

    def __call__(self, data):
        r"""
        Partition graph data by Louvain algorithm and similar nodes (i.e. being of the same community) will be
        allocated one client.
        Args:
            data (networkx.Graph):
        Returns:
            local_nodes (List): the local nodes id owned by each client (e.g. [[1,2], [3,4]])
        """
        local_nodes = [[] for _ in range(self.num_clients)]
        self.node_groups = community.community_louvain.best_partition(data)
        groups = collections.defaultdict(list)
        for ni, gi in self.node_groups.items():
            groups[gi].append(ni)
        groups = {k: groups[k] for k in list(range(len(groups)))}
        # ensure the number of groups is larger than the number of clients
        while len(groups) < self.num_clients:
            # find the group with the largest size
            groups_lens = [groups[k] for k in range(len(groups))]
            max_gi = np.argmax(groups_lens)
            # set the size of the new group
            min_glen = min(groups_lens)
            max_glen = max(groups_lens)
            if max_glen < 2 * min_glen:
                min_glen = max_glen // 2
            # split the group with the largest size into two groups
            nodes_in_gi = groups[max_gi]
            new_group_id = len(groups)
            groups[new_group_id] = nodes_in_gi[:min_glen]
            groups[max_gi] = nodes_in_gi[min_glen:]
        # allocate different groups to clients
        groups_lens = [len(groups[k]) for k in range(len(groups))]
        group_ids = np.argsort(groups_lens)
        for gi in group_ids:
            cid = np.argmin([len(li) for li in local_nodes])
            local_nodes[cid].extend(groups[gi])
        return local_nodes


class NodeMetisPartitioner(BasicPartitioner):
    """
    Partition a graph into several subgraph by louvain algorithms. The input
    of this partitioner should be of type networkx.Graph
    """

    def __init__(self, num_clients=100):
        super(NodeMetisPartitioner, self).__init__()
        self.num_clients = num_clients

    def __str__(self):
        name = "Metis"
        return name

    def __call__(self, data):
        r"""
        Partition graph data by Metis algorithm and similar nodes (i.e. being of the same community) will be
        allocated one client.
        Args:
            data (networkx.Graph):
        Returns:
            local_nodes (List): the local nodes id owned by each client (e.g. [[1,2], [3,4]])
        """
        n_cuts, membership = metis.part_graph(data, self.num_clients)
        local_nodes = []
        for i in range(self.num_clients):
            client_indices = np.where(np.array(membership) == i)[0]
            local_nodes.append(client_indices.tolist())
        return local_nodes


class NodeOverlappingPartitioner(BasicPartitioner):
    """
    Partition a graph into several subgraph by louvain algorithms. The input
    of this partitioner should be of type networkx.Graph
    from: ```Personalized Subgraph Federated Learning```
    """

    def __init__(self, num_clients=100):
        super(NodeOverlappingPartitioner, self).__init__()
        self.num_clients = num_clients

    def __str__(self):
        name = "Overlapping"
        return name

    def __call__(self, data):
        r"""
        Partition graph data by Metis algorithm and similar nodes (i.e. being of the same community) will be
        allocated one client.
        Args:
            data (networkx.Graph):
        Returns:
            local_nodes (List): the local nodes id owned by each client (e.g. [[1,2], [3,4]])
        """
        num_parts = self.num_clients // 5
        n_cuts, membership = metis.part_graph(data, num_parts)
        local_nodes = []

        for i in range(self.num_clients):
            mem_id = i % 2
            client_indices = np.where(np.array(membership) == mem_id)[0]
            d_idxs = np.random.permutation(len(client_indices))
            local_nodes.append(client_indices[d_idxs[:len(client_indices) // 2]].tolist())
        return local_nodes


class OneClassPartitioner(BasicPartitioner):
    """
    Partition a graph into several subgraph by One-Class algorithms. The input
    of this partitioner should be of type networkx.Graph
    """

    def __init__(self, num_clients=100, index_func=lambda X: [xi[-1] for xi in X], target_class=0):
        super(OneClassPartitioner, self).__init__()
        self.num_clients = num_clients
        self.index_func = index_func
        self.target_class = target_class

    def __str__(self):
        name = "One-Class"
        return name

    def __call__(self, data):
        r"""
        Partition graph data by nodes with a same class will be
        allocated one client.
        Args:
            data (networkx.Graph):
        Returns:
            local_nodes (List): the local nodes id owned by each client (e.g. [[1,2], [3,4]])
        """
        num_parts = self.num_clients
        attrs = self.index_func(data)
        num_attrs = len(set(attrs))

        lb_counter = collections.Counter(attrs)
        lb_names = list(lb_counter.keys())
        lb_dict = {}
        attrs = np.array(attrs)
        for lb in lb_names:
            lb_dict[lb] = np.where(attrs == lb)[0]

        client2labels = []
        for i in range(self.num_clients):
            lb = random.sample(list(range(num_attrs)), 1)
            while len(lb_dict[lb[0]]) < self.num_clients * self.num_clients or lb[0] == self.target_class:
                lb = random.sample(list(range(num_attrs)), 1)
            client2labels.append(lb)

        lb2count = [0 for _ in range(num_attrs)]
        for i in range(self.num_clients):
            lb2count[client2labels[i][0]] += 1

        local_nodes = []
        lb2idx = [0 for _ in range(num_attrs)]
        for i in range(self.num_clients):
            node_list = []
            lb1 = client2labels[i][0]
            num_samples_per_batch = len(lb_dict[lb1]) // lb2count[lb1]
            left, right = int(lb2idx[lb1] * num_samples_per_batch), \
                int((lb2idx[lb1] + 1) * num_samples_per_batch)
            node_list.extend(lb_dict[lb1][left: right].tolist())

            lb2idx[lb1] += 1
            local_nodes.append(node_list)
        return local_nodes


class TwoClassPartitioner(BasicPartitioner):
    """
    Partition a graph into several subgraph by Two-Class algorithms. The input
    of this partitioner should be of type networkx.Graph
    """

    def __init__(self, num_clients=100, index_func=lambda X: [xi[-1] for xi in X]):
        super(TwoClassPartitioner, self).__init__()
        self.num_clients = num_clients
        self.index_func = index_func

    def __str__(self):
        name = "Two-Class"
        return name

    def __call__(self, data):
        r"""
        Partition graph data by nodes with a same class will be
        allocated one client.
        Args:
            data (networkx.Graph):
        Returns:
            local_nodes (List): the local nodes id owned by each client (e.g. [[1,2], [3,4]])
        """
        num_parts = self.num_clients
        attrs = self.index_func(data)
        num_attrs = len(set(attrs))

        lb_counter = collections.Counter(attrs)
        lb_names = list(lb_counter.keys())
        lb_dict = {}
        attrs = np.array(attrs)
        for lb in lb_names:
            lb_dict[lb] = np.where(attrs == lb)[0]

        client2labels = []
        for i in range(self.num_clients):
            lb = random.sample(list(range(num_attrs)), 2)
            while len(lb_dict[lb[0]]) < self.num_clients or len(lb_dict[lb[1]]) < self.num_clients:
                lb = random.sample(list(range(num_attrs)), 2)
            client2labels.append(lb)

        lb2count = [0 for _ in range(num_attrs)]
        for i in range(self.num_clients):
            lb2count[client2labels[i][0]] += 1
            lb2count[client2labels[i][1]] += 1

        local_nodes = []
        lb2idx = [0 for _ in range(num_attrs)]
        for i in range(self.num_clients):
            node_list = []
            lb1 = client2labels[i][0]
            num_samples_per_batch = len(lb_dict[lb1]) // lb2count[lb1]
            left, right = int(lb2idx[lb1] * num_samples_per_batch), \
                int((lb2idx[lb1] + 1) * num_samples_per_batch)
            node_list.extend(lb_dict[lb1][left: right].tolist())

            lb2 = client2labels[i][1]
            num_samples_per_batch = len(lb_dict[lb2]) // lb2count[lb2]
            left, right = int(lb2idx[lb2] * num_samples_per_batch), \
                int((lb2idx[lb2] + 1) * num_samples_per_batch)
            node_list.extend(lb_dict[lb2][left: right].tolist())

            lb2idx[lb1] += 1
            lb2idx[lb2] += 1
            local_nodes.append(node_list)
        return local_nodes


class InterClientNonIIDPartitioner(BasicPartitioner):
    """
    Partition a graph into several subgraph by Inter-Client non-iid algorithms. The input
    of this partitioner should be of type networkx.Graph
    """

    def __init__(self, num_clients=100, index_func=lambda X: [xi[-1] for xi in X]):
        super(InterClientNonIIDPartitioner, self).__init__()
        self.num_clients = num_clients
        self.index_func = index_func

    def __str__(self):
        name = "Inter-Client non-iid"
        return name

    def __call__(self, data):
        r"""
        Partition graph data by nodes with a same class will be
        allocated one client.
        Args:
            data (networkx.Graph):
        Returns:
            local_nodes (List): the local nodes id owned by each client (e.g. [[1,2], [3,4]])
        """
        num_parts = self.num_clients
        attrs = self.index_func(data)
        num_attrs = len(set(attrs))

        lb_counter = collections.Counter(attrs)
        lb_names = list(lb_counter.keys())
        lb_dict = {}
        attrs = np.array(attrs)
        for lb in lb_names:
            lb_dict[lb] = np.where(attrs == lb)[0]

        client2labels = []
        for i in range(self.num_clients):
            elements = [0, 1]
            coin_flips = np.random.choice(elements, num_attrs, p=[0.5, 0.5])
            client2labels.append([lb for lb in range(num_attrs) if coin_flips[lb] == 1])
            if len(client2labels[i]) == 0:
                client2labels[i].append(random.choice(range(num_attrs)))

        client2ratio = []
        for i in range(self.num_clients):
            lb_ratio = []
            for _ in client2labels[i]:
                lb_ratio.append(random.uniform(0, 1))
            client2ratio.append(lb_ratio)

        lb2count = [0 for _ in range(num_attrs)]
        for i in range(self.num_clients):
            for idx, lb in enumerate(client2labels[i]):
                lb2count[lb] += client2ratio[i][idx]

        for i in range(self.num_clients):
            for idx, lb in enumerate(client2labels[i]):
                client2ratio[i][idx] /= lb2count[lb]

        local_nodes = []
        lb2idx = [0 for _ in range(num_attrs)]
        for i in range(self.num_clients):

            node_list = []
            for idx, lb in enumerate(client2labels[i]):
                ratio = client2ratio[i][idx]
                num_samples = int(len(lb_dict[lb]) * ratio)
                left, right = lb2idx[lb], lb2idx[lb] + num_samples
                node_list.extend(lb_dict[lb][left: right].tolist())
                lb2idx[lb] = right

            local_nodes.append(node_list)
        return local_nodes
