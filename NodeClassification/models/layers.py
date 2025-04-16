from typing import Any, Optional, Union, Tuple
import copy
import math

import torch
import torch.nn.functional as F
from torch.nn import Parameter
from torch import Tensor, nn

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense import Linear
from torch_geometric.typing import (
    Adj,
    OptPairTensor,
    OptTensor,
    SparseTensor,
    torch_sparse
)
from torch_geometric.nn.inits import zeros, glorot
from torch_geometric.utils import (
    spmm,
    remove_self_loops,
    is_torch_sparse_tensor,
    softmax
)
from torch_geometric.utils import add_self_loops as add_self_loops_fn
from torch_geometric.nn import inits
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils.sparse import set_sparse_value


def is_uninitialized_parameter(x: Any) -> bool:
    if not hasattr(nn.parameter, 'UninitializedParameter'):
        return False
    return isinstance(x, nn.parameter.UninitializedParameter)


class MaskedGCNLinear(torch.nn.Module):

    def __init__(self, in_channels: int, out_channels: int,
                 bias: bool = True, weight_initializer: Optional[str] = None, bias_initializer: Optional[str] = None, l1=1e-3):
        super(MaskedGCNLinear, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer
        self.l1 = l1

        if in_channels > 0:
            self.weight = Parameter(torch.Tensor(out_channels, in_channels))
        else:
            self.weight = nn.parameter.UninitializedParameter()
            self._hook = self.register_forward_pre_hook(self.initialize_parameters)

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self._load_hook = self._register_load_state_dict_pre_hook(self._lazy_load_hook)

        self.reset_parameters()

    def __deepcopy__(self, memo):
        out = MaskedGCNLinear(self.in_channels, self.out_channels, self.bias is not None, self.weight_initializer, self.bias_initializer)
        if self.in_channels > 0:
            out.weight = copy.deepcopy(self.weight, memo)
        if self.bias is not None:
            out.bias = copy.deepcopy(self.bias, memo)
        return out

    def reset_parameters(self):
        if self.in_channels <= 0:
            pass
        elif self.weight_initializer == 'glorot':
            inits.glorot(self.weight)
        elif self.weight_initializer == 'uniform':
            bound = 1.0 / math.sqrt(self.weight.size(-1))
            torch.nn.init.uniform_(self.weight.data, -bound, bound)
        elif self.weight_initializer == 'kaiming_uniform':
            inits.kaiming_uniform(self.weight, fan=self.in_channels, a=math.sqrt(5))
        elif self.weight_initializer is None:
            inits.kaiming_uniform(self.weight, fan=self.in_channels, a=math.sqrt(5))
        else:
            raise RuntimeError(f"Linear layer weight initializer "
                               f"'{self.weight_initializer}' is not supported")

        if self.bias is None or self.in_channels <= 0:
            pass
        elif self.bias_initializer == 'zeros':
            inits.zeros(self.bias)
        elif self.bias_initializer is None:
            inits.uniform(self.in_channels, self.bias)
        else:
            raise RuntimeError(f"Linear layer bias initializer "
                               f"'{self.bias_initializer}' is not supported")

    #####################################################
    def prune(self, mask):
        if self.training:
            return mask
        else:
            pruned = torch.abs(mask) < self.l1
            return mask.masked_fill(pruned, 0)
    #####################################################

    def forward(self, x: Tensor, m: Tensor) -> Tensor:
        r"""
        Args:
            x (Tensor): The features.
        """
        ##############################
        w = self.weight * self.prune(m)
        ##############################
        return F.linear(x, w, self.bias)

    @torch.no_grad()
    def initialize_parameters(self, module, input):
        if is_uninitialized_parameter(self.weight):
            self.in_channels = input[0].size(-1)
            self.weight.materialize((self.out_channels, self.in_channels))
            self.reset_parameters()
        self._hook.remove()
        delattr(self, '_hook')

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        if is_uninitialized_parameter(self.weight):
            destination[prefix + 'weight'] = self.weight
        else:
            destination[prefix + 'weight'] = self.weight.detach()
        if self.bias is not None:
            destination[prefix + 'bias'] = self.bias.detach()

    def _lazy_load_hook(self, state_dict, prefix, local_metadata, strict,
                        missing_keys, unexpected_keys, error_msgs):

        weight = state_dict[prefix + 'weight']
        if is_uninitialized_parameter(weight):
            self.in_channels = -1
            self.weight = nn.parameter.UninitializedParameter()
            if not hasattr(self, '_hook'):
                self._hook = self.register_forward_pre_hook(
                    self.initialize_parameters)

        elif is_uninitialized_parameter(self.weight):
            self.in_channels = weight.size(-1)
            self.weight.materialize((self.out_channels, self.in_channels))
            if hasattr(self, '_hook'):
                self._hook.remove()
                delattr(self, '_hook')

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, bias={self.bias is not None})')


class MaskedLinear(torch.nn.Module):

    def __init__(self, d_i, d_o, l1=1e-3, mask_one_init=False):
        super(MaskedLinear, self).__init__()
        self.d_i = d_i
        self.d_o = d_o
        self.l1 = l1
        self.mask_one_init = mask_one_init

        self.weight = torch.nn.Parameter(torch.empty((self.d_o, self.d_i), requires_grad=True, dtype=torch.float32))
        torch.nn.init.xavier_uniform_(self.weight)
        self.mask = torch.nn.Parameter(torch.ones((self.d_o, self.d_i), requires_grad=True, dtype=torch.float32))

        if not self.mask_one_init:
            torch.nn.init.xavier_uniform_(self.mask)

        self.bias = torch.nn.Parameter(torch.zeros((1, self.d_o), requires_grad=True, dtype=torch.float32))
        torch.nn.init.xavier_uniform_(self.bias)

    def set_mask(self):
        return self.mask

    def prune(self, mask):
        if self.training:
            return mask
        else:
            pruned = torch.abs(mask) < self.l1
            return mask.masked_fill(pruned, 0)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        curr_mask = self.set_mask()
        weight = self.weight * self.prune(curr_mask)
        return F.linear(input, weight, self.bias)


class MaskedGCNConv(MessagePassing):

    _cached_edge_index: Optional[OptPairTensor]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self, in_channels, out_channels, improved=False, cached=False,
                 add_self_loops=False, normalize=True, bias=True, l1=1e-3, mask_one_init=True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        self._cached_edge_index = None
        self._cached_adj_t = None

        self.l1 = l1
        self.mask_one_init = mask_one_init

        self.lin = MaskedGCNLinear(in_channels, out_channels, bias=False, l1=l1, weight_initializer='glorot')

        self.d_o, self.d_i = self.lin.weight.size()
        self.mask = torch.nn.Parameter(torch.ones((self.d_o, self.d_i), requires_grad=True, dtype=torch.float32))

        if not self.mask_one_init:
            torch.nn.init.xavier_uniform_(self.mask)

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin.reset_parameters()
        zeros(self.bias)
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:

        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, self.flow, x.dtype)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, self.flow, x.dtype)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        curr_mask = self.set_mask()
        x = self.lin(x, curr_mask)

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None)

        if self.bias is not None:
            out = out + self.bias

        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)

    def set_mask(self):
        return self.mask


class MaskedGATConv(MessagePassing):

    def __init__(self,
                 in_channels: Union[int, Tuple[int, int]],
                 out_channels: int,
                 heads: int = 1,
                 concat: bool = True,
                 negative_slope: float = 0.2,
                 dropout: float = 0.0,
                 add_self_loops: bool = True,
                 edge_dim: Optional[int] = None,
                 fill_value: Union[float, Tensor, str] = 'mean',
                 bias: bool = True,
                 l1: float = 1e-3,
                 mask_one_init: bool = True,
                 **kwargs):

        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.edge_dim = edge_dim
        self.fill_value = fill_value

        # In case we are operating in bipartite graphs, we apply separate
        # transformations 'lin_src' and 'lin_dst' to source and target nodes:
        self.l1 = l1
        self.mask_one_init = mask_one_init
        if isinstance(in_channels, int):
            self.lin_src = MaskedGCNLinear(in_channels, heads * out_channels, bias=False, weight_initializer='glorot')
            self.lin_dst = self.lin_src

        # The learnable parameters to compute attention coefficients:
        self.att_src = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_dst = Parameter(torch.Tensor(1, heads, out_channels))

        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False, weight_initializer='glorot')
            self.att_edge = Parameter(torch.Tensor(1, heads, out_channels))
        else:
            self.lin_edge = None
            self.register_parameter('att_edge', None)

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.d_o, self.d_i = self.lin.weight.size()
        self.mask = torch.nn.Parameter(torch.ones((self.d_o, self.d_i), requires_grad=True, dtype=torch.float32))

        if not self.mask_one_init:
            torch.nn.init.xavier_uniform_(self.mask)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin_src.reset_parameters()
        self.lin_dst.reset_parameters()
        if self.lin_edge is not None:
            self.lin_edge.reset_parameters()
        glorot(self.att_src)
        glorot(self.att_dst)
        glorot(self.att_edge)
        zeros(self.bias)

    def set_mask(self):
        return self.mask

    def forward(self, x, edge_index, edge_attr=None, size=None, return_attention_weights=None):

        H, C = self.heads, self.out_channels
        curr_mask = self.set_mask()

        # We first transform the input node features. If a tuple is passed, we
        # transform source and target node features via separate weights:
        if isinstance(x, Tensor):
            assert x.dim() == 2, "Static graphs not supported in 'GATConv'"
            x_src = x_dst = self.lin_src(x, curr_mask).view(-1, H, C)
        else:  # Tuple of source and target node features:
            x_src, x_dst = x
            assert x_src.dim() == 2, "Static graphs not supported in 'GATConv'"
            x_src = self.lin_src(x_src, curr_mask).view(-1, H, C)
            if x_dst is not None:
                x_dst = self.lin_dst(x_dst, curr_mask).view(-1, H, C)

        x = (x_src, x_dst)

        # Next, we compute node-level attention coefficients, both for source
        # and target nodes (if present):
        alpha_src = (x_src * self.att_src).sum(dim=-1)
        alpha_dst = None if x_dst is None else (x_dst * self.att_dst).sum(-1)
        alpha = (alpha_src, alpha_dst)

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                # We only want to add self-loops for nodes that appear both as
                # source and target nodes:
                num_nodes = x_src.size(0)
                if x_dst is not None:
                    num_nodes = min(num_nodes, x_dst.size(0))
                num_nodes = min(size) if size is not None else num_nodes
                edge_index, edge_attr = remove_self_loops(
                    edge_index, edge_attr)
                edge_index, edge_attr = add_self_loops_fn(
                    edge_index, edge_attr, fill_value=self.fill_value,
                    num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                if self.edge_dim is None:
                    edge_index = torch_sparse.set_diag(edge_index)
                else:
                    raise NotImplementedError(
                        "The usage of 'edge_attr' and 'add_self_loops' "
                        "simultaneously is currently not yet supported for "
                        "'edge_index' in a 'SparseTensor' form")

        # edge_updater_type: (alpha: OptPairTensor, edge_attr: OptTensor)
        alpha = self.edge_updater(edge_index, alpha=alpha, edge_attr=edge_attr)

        # propagate_type: (x: OptPairTensor, alpha: Tensor)
        out = self.propagate(edge_index, x=x, alpha=alpha, size=size)

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias

        if isinstance(return_attention_weights, bool):
            if isinstance(edge_index, Tensor):
                if is_torch_sparse_tensor(edge_index):
                    # TODO TorchScript requires to return a tuple
                    adj = set_sparse_value(edge_index, alpha)
                    return out, (adj, alpha)
                else:
                    return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def edge_update(self, alpha_j: Tensor, alpha_i: OptTensor,
                    edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
                    size_i: Optional[int]) -> Tensor:
        # Given edge-level attention coefficients for source and target nodes,
        # we simply need to sum them up to "emulate" concatenation:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        if index.numel() == 0:
            return alpha
        if edge_attr is not None and self.lin_edge is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            edge_attr = self.lin_edge(edge_attr)
            edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
            alpha_edge = (edge_attr * self.att_edge).sum(dim=-1)
            alpha = alpha + alpha_edge

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return alpha

    def message(self, x_j: Tensor, alpha: Tensor) -> Tensor:
        return alpha.unsqueeze(-1) * x_j

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')


class MaskedGraphConv(MessagePassing):

    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, aggr: str = 'add', bias: bool = True, l1: float = 1e-3, mask_one_init: bool = True, **kwargs):
        super().__init__(aggr=aggr, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.l1 = l1
        self.mask_one_init = mask_one_init

        self.lin_rel = MaskedGCNLinear(in_channels[0], out_channels, bias=False, l1=l1, weight_initializer='glorot')
        self.lin_root = MaskedGCNLinear(in_channels[1], out_channels, bias=False, l1=l1, weight_initializer='glorot')

        self.d_o, self.d_i = self.lin.weight.size()
        self.mask = torch.nn.Parameter(torch.ones((self.d_o, self.d_i), requires_grad=True, dtype=torch.float32))

        if not self.mask_one_init:
            torch.nn.init.xavier_uniform_(self.mask)

        self.reset_parameters()

    def set_mask(self):
        return self.mask

    def forward(self, x, edge_index, edge_attr=None, size=None):
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        curr_mask = self.set_mask()

        # propagate_type: (x: OptPairTensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_attr, size=size)
        out = self.lin_rel(out, curr_mask)

        x_r = x[1]
        if x_r is not None:
            out = out + self.lin_root(x_r, curr_mask)

        return out

    def reset_parameters(self):
        super().reset_parameters()
        self.lin_rel.reset_parameters()
        self.lin_root.reset_parameters()

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: OptPairTensor) -> Tensor:
        return spmm(adj_t, x[0], reduce=self.aggr)


class MaskedSGConv(MessagePassing):
    _cached_x: Optional[Tensor]

    def __init__(self, in_channels: int, out_channels: int,
                 K: int = 1, cached: bool = False, add_self_loops: bool = True,
                 bias: bool = True, l1: float = 1e-3, mask_one_init: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.cached = cached
        self.add_self_loops = add_self_loops

        self._cached_x = None
        self.l1 = l1
        self.mask_one_init = mask_one_init
        self.lin = MaskedGCNLinear(in_channels, out_channels, bias=False, l1=l1, weight_initializer='glorot')

        self.d_o, self.d_i = self.lin.weight.size()
        self.mask = torch.nn.Parameter(torch.ones((self.d_o, self.d_i), requires_grad=True, dtype=torch.float32))
        if not self.mask_one_init:
            torch.nn.init.xavier_uniform_(self.mask)
        self.reset_parameters()

    def set_mask(self):
        return self.mask

    def reset_parameters(self):
        super().reset_parameters()
        self.lin.reset_parameters()
        self._cached_x = None

    def forward(self, x, edge_index, edge_attr=None):
        curr_mask = self.set_mask()

        cache = self._cached_x
        if cache is None:
            if isinstance(edge_index, Tensor):
                edge_index, edge_attr = gcn_norm(  # yapf: disable
                    edge_index, edge_attr, x.size(self.node_dim), False,
                    self.add_self_loops, self.flow, dtype=x.dtype)
            elif isinstance(edge_index, SparseTensor):
                edge_index = gcn_norm(  # yapf: disable
                    edge_index, edge_attr, x.size(self.node_dim), False,
                    self.add_self_loops, self.flow, dtype=x.dtype)

            for k in range(self.K):
                # propagate_type: (x: Tensor, edge_weight: OptTensor)
                x = self.propagate(edge_index, x=x, edge_weight=edge_attr, size=None)
                if self.cached:
                    self._cached_x = x
        else:
            x = cache.detach()

        return self.lin(x, curr_mask)

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, K={self.K})')
