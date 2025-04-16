from models.appnp import APPNP
from models.elasticgnn import ElasticGNN
from models.encoder import GCN_Encoder
from models.gat import GAT
from models.gcn import GCN
from models.gnnguard import GNNGuard
from models.graphsage import GraphSage
from models.mediangcn import MedianGCN
from models.robustgcn import RobustGCN
from models.sgc import SGC


def model_construct(data, num_hidden, num_classes, dropout, model_name, num_layer=2, masked=False, l1=1e-3, mask_one_init=True):

    if model_name == 'GCN':
        model = GCN(num_feat=data.x.shape[1], num_hidden=num_hidden,
                    num_classes=num_classes, dropout=dropout, masked=masked, l1=l1, mask_one_init=mask_one_init)
    elif model_name == 'GAT':
        model = GAT(num_feat=data.x.shape[1], num_hidden=num_hidden,
                    num_classes=num_classes, dropout=dropout, masked=masked, l1=l1, mask_one_init=mask_one_init)
    elif model_name == 'GraphSage':
        model = GraphSage(num_feat=data.x.shape[1], num_hidden=num_hidden,
                          num_classes=num_classes, dropout=dropout, masked=masked, l1=l1, mask_one_init=mask_one_init)
    elif model_name == 'APPNP':
        model = APPNP(n_feat=data.x.shape[1], n_hid=num_hidden, n_class=num_classes, num_layers=num_layer, dropout=dropout)
    elif model_name == 'GCN_Encoder':
        model = GCN_Encoder(num_feat=data.x.shape[1], num_hidden=num_hidden,
                            num_classes=num_classes, layer=num_layer, dropout=dropout)
    elif model_name == 'SGC':
        model = SGC(num_feat=data.x.shape[1], num_hidden=num_hidden,
                    num_classes=num_classes, dropout=dropout, masked=masked, l1=l1, mask_one_init=mask_one_init)
    elif model_name == 'GNNGuard':
        model = GNNGuard(n_feat=data.x.shape[1], n_hid=num_hidden, n_class=num_classes, use_ln=True, dropout=dropout)
    elif model_name == 'RobustGCN':
        model = RobustGCN(n_feat=data.x.shape[1], n_hid=num_hidden, n_class=num_classes, num_layers=num_layer, dropout=dropout)
    elif model_name == 'MedianGCN':
        model = MedianGCN(n_feat=data.x.shape[1], n_hid=num_hidden, n_class=num_classes, num_layers=num_layer, dropout=dropout)
    elif model_name == 'ElasticGNN':
        model = ElasticGNN(n_feat=data.x.shape[1], n_hid=num_hidden, n_class=num_classes, num_layers=num_layer, dropout=dropout)
    else:
        raise ValueError('Invalid model name')
    return model
