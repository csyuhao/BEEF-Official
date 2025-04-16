import torch
import numpy as np


def amazon_data(dataset):

    graph = dataset[0]
    labels = graph.y.numpy()

    dev_size = int(labels.shape[0] * 0.1)
    test_size = int(labels.shape[0] * 0.8)

    perm = np.random.permutation(labels.shape[0])
    test_index = perm[:test_size]
    dev_index = perm[test_size:test_size + dev_size]

    data_index = np.arange(labels.shape[0])
    test_mask = torch.tensor(np.in1d(data_index, test_index), dtype=torch.bool)
    dev_mask = torch.tensor(np.in1d(data_index, dev_index), dtype=torch.bool)

    train_mask = ~(dev_mask + test_mask)
    test_mask = test_mask.reshape(1, -1).squeeze(0)
    val_mask = dev_mask.reshape(1, -1).squeeze(0)
    train_mask = train_mask.reshape(1, -1).squeeze(0)

    graph.train_mask = train_mask
    graph.val_mask = val_mask
    graph.test_mask = test_mask
    return graph
