import os
import random
import logging
import numpy as np
from copy import deepcopy

import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn.functional as F

import dgl
import torch_geometric as torch_geo

try:
    if 'logger' not in globals():
        logging.basicConfig()
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
except NameError:
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)


def seed_experiment(seed=0):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch_geo.seed_everything(seed)
    dgl.random.seed(seed)

    cudnn.deterministic = True
    cudnn.benchmark = False
    logger.info('Seeded everything')


def accuracy(output, labels):
    """Return accuracy of output compared to labels.
    Parameters
    ----------
    output : torch.Tensor
        output from model
    labels : torch.Tensor or numpy.array
        node labels
    Returns
    -------
    float
        accuracy
    """

    if not hasattr(labels, '__len__'):
        labels = [labels]

    if type(labels) is not torch.Tensor:
        labels = torch.LongTensor(labels)

    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()

    return correct / len(labels)


def fit_model(model, lr, weight_decay, train_epochs, data, train_edge_index, idx_train, idx_val, device):
    data, idx_train, idx_val, train_edge_index = data.to(device), idx_train.to(device), idx_val.to(device), train_edge_index.to(device)
    edge_weight = torch.ones([data.edge_index.shape[1]], device=device, dtype=torch.float32)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_loss_val, best_acc_val, best_state_dict = 100, 0., None
    for i in range(train_epochs):
        model.train()
        optimizer.zero_grad()
        output = model(data.x, train_edge_index, edge_weight)
        loss_train = F.cross_entropy(output[idx_train], data.y[idx_train])
        loss_train.backward()
        optimizer.step()

        model.eval()
        output = model(data.x, train_edge_index, edge_weight)
        acc_val = accuracy(output[idx_val], data.y[idx_val])

        if acc_val > best_acc_val:
            best_acc_val = acc_val
            best_state_dict = deepcopy(model.state_dict())

    model.load_state_dict(best_state_dict)
    return model


@torch.no_grad()
def eval_model(model, data, idx_test, device):
    model.eval()
    data, idx_test = data.to(device), idx_test.to(device)
    edge_weight = torch.ones([data.edge_index.shape[1]], device=device, dtype=torch.float32)

    with torch.no_grad():
        output = model(data.x, data.edge_index.to(device), edge_weight)
        acc_test = accuracy(output[idx_test], data.y[idx_test].reshape(-1))
    return float(acc_test)
