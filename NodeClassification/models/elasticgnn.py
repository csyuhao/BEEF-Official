from typing import Optional, Union, Tuple

import torch
from torch import Tensor, nn
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.nn.conv.gcn_conv import gcn_norm

from torch_geometric.utils import (
    degree,
    scatter,
    sort_edge_index,
    to_dense_batch,
    add_self_loops,
)
from torch_sparse import SparseTensor, mul, matmul, fill_diag


def spmm(x: Tensor, edge_index: Union[Tensor, SparseTensor],
         edge_weight: OptTensor = None, reduce: str = 'sum') -> Tensor:
    r"""Sparse-dense matrix multiplication.

    Parameters
    ----------
    x : torch.Tensor
        the input dense 2D-matrix
    edge_index : torch.Tensor
        the location of the non-zeros elements in the sparse matrix,
        denoted as :obj:`edge_index` with shape [2, M]
    edge_weight : Optional[Tensor], optional
        the edge weight of the sparse matrix, by default None
    reduce : str, optional
        reduction of the sparse matrix multiplication, including:
        (:obj:`'mean'`, :obj:`'sum'`, :obj:`'add'`,
        :obj:`'max'`, :obj:`'min'`, :obj:`'median'`,
        :obj:`'sample_median'`)
        by default :obj:`'sum'`

    Returns
    -------
    Tensor
        the output result of the matrix multiplication.

    Example
    -------
    .. code-block:: python

        import torch
        from greatx.functional import spmm

        x = torch.randn(5, 2)
        edge_index = torch.LongTensor([[1,2], [3,4]])
        out1 = spmm(x, edge_index, reduce='sum')

        # which is equivalent to:
        A = torch.zeros(5, 5)
        A[edge_index[0], edge_index[1]] = 1.0
        out2 = torch.mm(A.t(), x)

        assert torch.allclose(out1, out2)

        # Also, it also supports :obj:`torch.sparse.Tensor`
        # and :obj:`torch_sparse.SparseTensor`
        A = A.to_sparse()
        out3 = spmm(x, A.t())
        assert torch.allclose(out1, out3)

        A = SparseTensor.from_torch_sparse_coo_tensor(A)
        out4 = spmm(x, A.t())
        assert torch.allclose(out1, out4)

    See also
    --------
    :class:`~torch_geometric.utils.spmm` (>=2.2.0)
    """

    # Case 1: `torch_sparse.SparseTensor`
    if isinstance(edge_index, SparseTensor):
        assert reduce in ['sum', 'add', 'mean', 'min', 'max']
        return matmul(edge_index, x, reduce)

    # Case 2: `torch.sparse.Tensor` (Sparse) and `torch.FloatTensor` (Dense)
    if isinstance(edge_index, Tensor) and (edge_index.is_sparse
                                           or edge_index.dtype == torch.float):
        assert reduce in ['sum', 'add']
        return torch.sparse.mm(edge_index, x)

    # Case 3: `torch.LongTensor` (Sparse)
    if reduce == 'median':
        return scatter_median(x, edge_index, edge_weight)
    elif reduce == 'sample_median':
        return scatter_sample_median(x, edge_index, edge_weight)

    row, col = edge_index
    x = x if x.dim() > 1 else x.unsqueeze(-1)

    out = x[row]
    if edge_weight is not None:
        out = out * edge_weight.unsqueeze(-1)
    out = scatter(out, col, dim=0, dim_size=x.size(0), reduce=reduce)
    return out


def scatter_median(x: Tensor, edge_index: Tensor,
                   edge_weight: OptTensor = None) -> Tensor:
    # NOTE: `to_dense_batch` requires the `index` is sorted by column
    ix = torch.argsort(edge_index[1])
    edge_index = edge_index[:, ix]
    row, col = edge_index
    x_j = x[row]

    if edge_weight is not None:
        x_j = x_j * edge_weight[ix].unsqueeze(-1)

    dense_x, mask = to_dense_batch(x_j, col, batch_size=x.size(0))
    h = x_j.new_zeros(dense_x.size(0), dense_x.size(-1))
    deg = mask.sum(dim=1)
    for i in deg.unique():
        if i == 0:
            continue
        deg_mask = deg == i
        h[deg_mask] = dense_x[deg_mask, :i].median(dim=1).values
    return h


def scatter_sample_median(x: Tensor, edge_index: Tensor,
                          edge_weight: OptTensor = None) -> Tensor:
    """Approximating the median aggregation with fixed set of
    neighborhood sampling."""

    try:
        from glcore import neighbor_sampler_cpu  # noqa
    except (ImportError, ModuleNotFoundError):
        raise ModuleNotFoundError(
            "`scatter_sample_median` requires glcore which "
            "is not installed, please refer to "
            "'https://github.com/EdisonLeeeee/glcore' "
            "for more information.")

    if edge_weight is not None:
        edge_index, edge_weight = sort_edge_index(edge_index, edge_weight,
                                                  sort_by_row=False)
    else:
        edge_index = sort_edge_index(edge_index, sort_by_row=False)

    row, col = edge_index
    num_nodes = x.size(0)
    deg = degree(col, dtype=torch.long, num_nodes=num_nodes)
    colptr = torch.cat([deg.new_zeros(1), deg.cumsum(dim=0)], dim=0)
    replace = True
    size = int(deg.float().mean().item())
    nodes = torch.arange(num_nodes)
    targets, neighbors, e_id = neighbor_sampler_cpu(colptr.cpu(), row.cpu(),
                                                    nodes, size, replace)

    x_j = x[neighbors]

    if edge_weight is not None:
        x_j = x_j * edge_weight[e_id].unsqueeze(-1)

    return x_j.view(num_nodes, size, -1).median(dim=1).values

def dense_add_self_loops(adj: Tensor, fill_value: float = 1.0) -> Tensor:
    diag = torch.diag(adj.new_full((adj.size(0), ), fill_value))
    return adj + diag


def dense_gcn_norm(adj: Tensor, improved: bool = False,
                   add_self_loops: bool = True, rate: float = -0.5) -> Tensor:
    fill_value = 2. if improved else 1.
    if add_self_loops:
        adj = dense_add_self_loops(adj, fill_value)
    deg = adj.sum(dim=1)
    deg_inv_sqrt = deg.pow_(rate)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
    norm_src = deg_inv_sqrt.view(1, -1)
    norm_dst = deg_inv_sqrt.view(-1, 1)
    adj = norm_src * adj * norm_dst
    return adj


def make_gcn_norm(
    edge_index: Adj,
    edge_weight: OptTensor = None,
    num_nodes: Optional[int] = None,
    add_self_loops: bool = True,
    dtype: Optional[torch.dtype] = None,
) -> Tuple[Adj, OptTensor]:
    r"""Perform GCN-normalization :math:`\mathbf{\hat{D}}^{-1/2}
    \mathbf{\hat{A}} \mathbf{\hat{D}}^{-1/2}`for input graph.

    Parameters
    ----------
    edge_index : Adj
        input graph denoted by `edge_index`, could be
        :obj:`torch.FloatTensor`,
        :obj:`torch_sparse.SparseTensor`,
        or :obj:`torch.LongTensor`.
    edge_weight : OptTensor, optional
        edge weights for the input edge_index, by default None
    num_nodes : Optional[int], optional
        number of nodes, by default None
    add_self_loops : bool, optional
        whether to add self-loop edges, by default True
    dtype : Optional[torch.dtype], optional
        types of edge weights of added self-loop edges
        if :obj:`add_self_loops=True`.

    Returns
    -------
    Tuple[Adj, OptTensor]
        output normalized graph denoted as
        :obj:`edge_index` and :obj:`edge_weight`.
    """
    if isinstance(edge_index, Tensor) and edge_index.dtype == torch.long:
        # Sparse edge_index with shape [2, M]
        edge_index, edge_weight = gcn_norm(edge_index, edge_weight,
                                           num_nodes=num_nodes, improved=False,
                                           add_self_loops=add_self_loops,
                                           dtype=dtype)
    elif isinstance(edge_index, Tensor) and edge_index.dtype == torch.float:
        # N by N dense adjacency matrix
        edge_index = dense_gcn_norm(edge_index, improved=False,
                                    add_self_loops=add_self_loops)
    elif isinstance(edge_index, SparseTensor):
        edge_index = gcn_norm(edge_index, num_nodes=num_nodes, improved=False,
                              add_self_loops=add_self_loops, dtype=dtype)
    else:
        raise ValueError(f"Type {type(edge_index)} is not supported.")

    return edge_index, edge_weight


def make_self_loops(
    edge_index: Adj,
    edge_weight: OptTensor = None,
    num_nodes: Optional[int] = None,
    fill_value: float = 1.0,
    improved: bool = False,
) -> Tuple[Adj, OptTensor]:
    r"""Add self-loop edges for input graph.

    Parameters
    ----------
    edge_index : Adj
        input graph denoted by `edge_index`, could be
        :obj:`torch.FloatTensor`,
        :obj:`torch_sparse.SparseTensor`,
        or :obj:`torch.LongTensor`.
    edge_weight : OptTensor, optional
        edge weights for the input edge_index, by default None
    num_nodes : Optional[int], optional
        number of nodes, by default None
    fill_value : float, optional
        fill value for the added self-loop edges,
        by default 1.0
    improved : bool, optional
        whether the layer computes
        :math:`\mathbf{\hat{A}}` as :math:`\mathbf{A} + 2\mathbf{I}`,
        by default False

    Returns
    -------
    Tuple[Adj, OptTensor]
        output edge indices and edge weights with
        added self-loop edges.
    """

    fill_value = 2. if improved else 1.
    if isinstance(edge_index, Tensor) and edge_index.dtype == torch.long:
        # Sparse edge_index with shape [2, M]
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight,
                                                 fill_value=fill_value,
                                                 num_nodes=num_nodes)
    elif isinstance(edge_index, Tensor) and edge_index.dtype == torch.float:
        # N by N dense adjacency matrix
        edge_index = dense_add_self_loops(edge_index, fill_value)
    elif isinstance(edge_index, SparseTensor):
        edge_index = fill_diag(edge_index, fill_value)
    else:
        raise ValueError(f"Type {type(edge_index)} is not supported.")

    return edge_index, edge_weight


def get_inc(edge_index: Adj, num_nodes: Optional[int] = None) -> SparseTensor:
    """Compute the incident matrix
    """
    device = edge_index.device
    if torch.is_tensor(edge_index):
        row_index, col_index = edge_index
        num_nodes = maybe_num_nodes(edge_index, num_nodes)
    else:
        row_index = edge_index.storage.row()
        col_index = edge_index.storage.col()
        num_nodes = edge_index.sizes()[1]

    mask = row_index > col_index  # remove duplicate edge and self loop

    row_index = row_index[mask]
    col_index = col_index[mask]
    num_edges = row_index.numel()

    row = torch.cat([
        torch.arange(num_edges, device=device),
        torch.arange(num_edges, device=device)
    ])
    col = torch.cat([row_index, col_index])
    value = torch.cat([
        torch.ones(num_edges, device=device),
        -torch.ones(num_edges, device=device)
    ])
    inc_mat = SparseTensor(row=row, rowptr=None, col=col, value=value,
                           sparse_sizes=(num_edges, num_nodes))
    return inc_mat


def inc_norm(inc: SparseTensor, edge_index: Adj,
             num_nodes: Optional[int] = None) -> SparseTensor:
    """Normalize the incident matrix
    """

    if torch.is_tensor(edge_index):
        deg = degree(edge_index[0], num_nodes=num_nodes,
                     dtype=torch.float).clamp(min=1)
    else:
        deg = edge_index.sum(1).clamp(min=1)

    deg_inv_sqrt = deg.pow(-0.5)
    inc = mul(inc, deg_inv_sqrt.view(1, -1))  # col-wise
    return inc


class ElasticConv(nn.Module):
    r"""
    The ElasticGNN operator from the `"Elastic Graph Neural
    Networks" <https://arxiv.org/abs/2107.06996>`_
    paper (ICML'21)

    Parameters
    ----------
    K : int, optional
        the number of propagation steps, by default 3
    lambda_amp : float, optional
        trade-off of adaptive message passing, by default 0.1
    normalize : bool, optional
        Whether to add self-loops and compute
        symmetric normalization coefficients on the fly, by default True
    add_self_loops : bool, optional
        whether to add self-loops to the input graph, by default True
    lambda1 : float, optional
        trade-off hyperparameter, by default 3
    lambda2 : float, optional
        trade-off hyperparameter, by default 3
    L21 : bool, optional
        whether to use row-wise projection
        on the l2 ball of radius λ1., by default True
    cached : bool, optional
        whether to cache the incident matrix, by default True


    See also
    --------
    :class:`greatx.nn.models.supervised.ElasticGNN`
    """

    _cached: Optional[SparseTensor] = None  # incident matrix

    def __init__(self, K: int = 3, lambda_amp: float = 0.1,
                 normalize: bool = True, add_self_loops: bool = True,
                 lambda1: float = 3., lambda2: float = 3., L21: bool = True,
                 cached: bool = True):

        super().__init__()

        self.K = K
        self.lambda_amp = lambda_amp
        self.add_self_loops = add_self_loops
        self.normalize = normalize
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.L21 = L21
        self.cached = cached

    def reset_parameters(self):
        self.cache_clear()

    def cache_clear(self):
        """Clear cached inputs or intermediate results."""
        self._cached = None
        return self

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        """"""

        cache = self._cached

        if cache is None:
            if self.add_self_loops:
                # NOTE: we do not support Dense adjacency matrix here
                edge_index, edge_weight = make_self_loops(
                    edge_index, edge_weight, num_nodes=x.size(0))

            if self.normalize:
                # NOTE: we do not support Dense adjacency matrix here
                edge_index, edge_weight = make_gcn_norm(
                    edge_index, edge_weight, num_nodes=x.size(0),
                    dtype=x.dtype, add_self_loops=False)

            # compute incident matrix before normalizing edge_index
            inc_mat = get_inc(edge_index, num_nodes=x.size(0))
            # normalize incident matrix
            inc_mat = inc_norm(inc_mat, edge_index, num_nodes=x.size(0))

            if self.cached:
                self._cached = (inc_mat, edge_index, edge_weight)
            self.init_z = x.new_zeros((inc_mat.sizes()[0], x.size()[-1]))
        else:
            inc_mat, edge_index, edge_weight = self._cached

        return self.emp_forward(x, inc_mat, edge_index, edge_weight)

    def emp_forward(self, x: Tensor, inc_mat: SparseTensor, edge_index: Adj, edge_weight: OptTensor = None) -> Tensor:
        lambda1 = self.lambda1
        lambda2 = self.lambda2

        gamma = 1 / (1 + lambda2)
        beta = 1 / (2 * gamma)

        hh = x

        if lambda1:
            z = self.init_z

        for k in range(self.K):

            if lambda2:
                out = spmm(x, edge_index, edge_weight)

                y = gamma * hh + (1 - gamma) * out
            else:
                y = gamma * hh + (1 - gamma) * x  # y = x - gamma * (x - hh)

            if lambda1:
                x_bar = y - gamma * (inc_mat.t() @ z)
                z_bar = z + beta * (inc_mat @ x_bar)
                if self.L21:
                    z = self.L21_projection(z_bar, lambda_=lambda1)
                else:
                    z = self.L1_projection(z_bar, lambda_=lambda1)
                x = y - gamma * (inc_mat.t() @ z)
            else:
                x = y  # z = 0

        return x

    def L1_projection(self, x: Tensor, lambda_: float) -> Tensor:
        """component-wise projection onto the l∞ ball of radius λ1."""
        return torch.clamp(x, min=-lambda_, max=lambda_)

    def L21_projection(self, x: Tensor, lambda_: float) -> Tensor:
        # row-wise projection on the l2 ball of radius λ1.
        row_norm = torch.norm(x, p=2, dim=1)
        scale = torch.clamp(row_norm, max=lambda_)
        index = row_norm > 0
        scale[index] = scale[index] / \
            row_norm[index]  # avoid to be devided by 0
        return scale.unsqueeze(1) * x

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(K={self.K})"


class ElasticGNN(torch.nn.Module):
    def __init__(self, n_feat, n_hid, n_class, num_layers, dropout):
        super(ElasticGNN, self).__init__()

        layers = []
        for idx in range(num_layers):
            if idx == 0:
                layers.append(nn.Linear(n_feat, n_hid, bias=True))
            elif idx == num_layers - 1:
                layers.append(nn.Linear(n_hid, n_class, bias=True))
            else:
                layers.append(nn.Linear(n_hid, n_hid, bias=True))

            if idx != num_layers - 1:
                layers.append(nn.LayerNorm(n_hid))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))

        layers.append(ElasticConv(K=3, lambda_amp=0.1, lambda1=3, lambda2=3, add_self_loops=True, cached=False))
        self.layers = nn.Sequential(*layers)

    def forward(self, x, edge_index, edge_weight=None, is_proxy=False):
        final_proxy = None
        for layer in self.layers:
            if isinstance(layer, ElasticConv):
                x = layer(x, edge_index, edge_weight)
            elif isinstance(layer, nn.Linear):
                x = layer(x)
                final_proxy = x
            else:
                x = layer(x)
        if is_proxy:
            return final_proxy
        return x
