from typing import Optional, Tuple, Dict, Union


import torch
from torch import Tensor
from torch_scatter import scatter
from torch_geometric.typing import Adj, OptTensor
from torch_sparse import SparseTensor, remove_diag
import torch_geometric.nn as tg
import torch_geometric.utils as tg_utils
import scipy.sparse as sp
import time

# TGP base classes
from tgp.src import SRCPooling
from tgp.connect import SparseConnect
from tgp.lift import BaseLift
from tgp.reduce import BaseReduce
from tgp.select import Select, SelectOutput
from tgp.src import PoolingOutput, SRCPooling
from tgp.utils.typing import ConnectionType, LiftType, ReduceType, SinvType
from tgp.utils import connectivity_to_edge_index, connectivity_to_sparse_tensor

# ---------------------------------------------------------------------------
# Two settings, deliberately fixed here rather than exposed as CLI flags.
#
# AUX_LOSS_WEIGHT
#   compute_loss divides by m^2, but ||W - M||^2 - ||M||^2 is itself O(m^2), so
#   the auxiliary loss lands at O(1) -- the same order as the classification
#   nll_loss.  At weight 1.0 the correlation-clustering objective is a second
#   task of equal weight rather than an auxiliary regulariser.
#
# DETACH_SELECTOR
#   the selector reads `x` straight out of the trained GIN trunk, so without
#   this the auxiliary gradient flows back through the selector into the trunk:
#   the CC objective would not merely train the selector, it would reshape the
#   shared graph representation and compete with classification.
# ---------------------------------------------------------------------------
AUX_LOSS_WEIGHT = 1.0
DETACH_SELECTOR = True




class CCSelector(Select):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        ratio: float,
        s_inv_op: SinvType = "transpose",
        linear: bool = False,
    ) -> None:
        super().__init__()

        self.pool_ratio = float(ratio)
        self.s_inv_op = s_inv_op
        self.linear = linear

        if self.linear:
            self.model = tg.Linear(in_channels, hidden_channels, bias=False)
        else:
            self.model = tg.GCNConv(in_channels, hidden_channels, bias=False)

    def forward(
        self,
        x: Tensor,
        adj: Adj,
        edge_weight: OptTensor = None,
        batch: Optional[Tensor] = None,
        **kwargs,
    ) -> SelectOutput:

        # Convert adjacency to edge_index COO for computations if needed
        if isinstance(adj, SparseTensor):
            row, col, _ = adj.coo()
            edge_index = torch.stack([row, col], dim=0)
        else:
            edge_index = adj

        num_nodes = x.size(0)

        # `x` here is the output of the trained GIN trunk, so the auxiliary CC
        # loss back-propagates through the selector INTO the shared trunk: it
        # does not merely train the selector, it pulls the graph representation
        # itself towards a correlation-clustering optimum, competing with the
        # classification loss.  DETACH_SELECTOR confines the CC objective to
        # the selector and leaves the trunk to the downstream task.
        x_sel = x.detach() if DETACH_SELECTOR else x

        # Embeddings
        if self.linear:
            # FIX: tg.Linear.forward accepts only x; passing edge_weight raised
            # TypeError, so the --linear_cc path could never actually run.
            h = self.model(x_sel)
        else:
            h = self.model(x_sel, edge_index, edge_weight=edge_weight)

        h = torch.nn.functional.normalize(h, p=2, dim=1)

        # Clustering and reduction
        cluster = self._make_cc_clusters(h, edge_index, batch)
        so = SelectOutput(
            cluster_index=cluster,
            num_nodes=num_nodes,
            num_supernodes=int(cluster.max().item()) + 1 if cluster.numel() > 0 else 0,
            h=h,
            s_inv_op=self.s_inv_op,
        )

        return so

    def _make_cc_clusters(self, out, edge_index, batch) -> Tensor:
        """
        Keep the (1 - pool_ratio) fraction of edges with the smallest embedding
        distance and return the connected components of what remains.

        """
        num_nodes = out.size(0)
        undirected_edge_index = edge_index[:, edge_index[0] < edge_index[1]]
        device = out.device

        if undirected_edge_index.size(1) == 0:
            return torch.arange(num_nodes, dtype=torch.long, device=device)

        row_out = out[undirected_edge_index[0]]
        col_out = out[undirected_edge_index[1]]
        dists_sq = torch.sum((row_out - col_out) ** 2, dim=1)

        k = max(int((1 - self.pool_ratio) * dists_sq.size(0)), 1)
        topk_vals, topk_indices = torch.topk(dists_sq, k, largest=False)
        filtered_edges = undirected_edge_index[:, topk_indices]

        filtered_edges = torch.cat(
            [filtered_edges, filtered_edges[[1, 0], :]], dim=1
        )  # make directed

        sp_graph = tg_utils.to_scipy_sparse_matrix(filtered_edges.cpu(), num_nodes=num_nodes)
        _, labels = sp.csgraph.connected_components(sp_graph, directed=False, return_labels=True)
        clustering = torch.from_numpy(labels).long().to(device)

        return clustering

    def reset_parameters(self):
        self.model.reset_parameters()

    def to(self, device):
        self.model.to(device)
        return self


class CCPooler(SRCPooling):
    """
    Learns a single GCN layer with the LinkGNN loss from `src/train.py`, then
    forms clusters by keeping the closest (1 - ratio) fraction of edges and
    taking connected components.

    The pooling reduces the input graph to cluster-level nodes by aggregating
    features per cluster (default: sum) and collapsing edges between clusters
    (default: sum), removing self-loops.

    Args:
        in_channels (int): Input feature dimension per node.
        hidden_channels (int): Dimension of the learned embedding (GCN output).
        ratio (float): Fraction of edges to drop when forming clusters.
        aggr_x (str): Feature aggregation over clusters (e.g., 'sum', 'mean', 'max').
        aggr_edge (str): Edge aggregation between clusters for SparseTensor coalesce.
        remove_self_loops (bool): Remove self-loops in the reduced graph.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        ratio: float,
        max_nodes: int = None,
        remove_self_loops: bool = True,
        precaching: bool = True,
        linear: bool = False,
        num_pivots: bool = None,
        lift: LiftType = "precomputed",
        s_inv_op: SinvType = "transpose",
        reduce_red_op: ReduceType = "sum",
        connect_red_op: ConnectionType = "sum",
        lift_red_op: ReduceType = "sum",
        degree_norm: bool = False,
        edge_weight_norm: bool = False,
    ) -> None:
        super().__init__(
            selector=CCSelector(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                ratio=ratio,
                s_inv_op=s_inv_op,
                linear=linear,
            ),
            reducer=BaseReduce(reduce_op=reduce_red_op),
            connector=SparseConnect(
                reduce_op=connect_red_op,
                edge_weight_norm=edge_weight_norm,
                degree_norm=degree_norm,
                remove_self_loops=remove_self_loops,
            ),
            lifter=BaseLift(matrix_op=lift, reduce_op=lift_red_op),
        )
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.pool_ratio = float(ratio)
        self.precaching = precaching
        self.linear = linear
        self.num_pivots = num_pivots
        # Precompute W matrix for aux loss computation if precaching is enabled
        if not self.precaching:
            self.W = None
        else:
            if max_nodes is None:
                raise ValueError("max_nodes must be provided if precaching is enabled.")
            self.W = torch.ones((max_nodes, max_nodes)) * (-1.0)
            self.W.fill_diagonal_(0.0)

    def _fresh_W(self, m: int, device) -> Tensor:
        """
        Return a clean m x m signed weight matrix: -1 off-diagonal, 0 on the diagonal.

        FIX: the previous code did `W = self.W[:m, :m]`, which is a *view* onto the
        cached buffer, and then wrote +1 into it at every edge position. Those +1
        entries were never cleared, so after a handful of graphs the cache held the
        accumulated union of every edge set seen so far and the auxiliary loss was
        being computed against a progressively more corrupted target -- silently,
        and getting worse the longer the run went on.
        """
        if self.W is None:
            W = torch.ones((m, m), device=device) * (-1.0)
        else:
            W = self.W[:m, :m]
            W.fill_(-1.0)
        W.fill_diagonal_(0.0)
        return W

    def compute_loss(self, out: Tensor, adj: Adj, edge_weight: OptTensor = None) -> Tensor:
        """Compute the LinkGNN-style loss on the current embeddings.

        Loss per graph: ||W - M||_F^2 - ||M||_F^2 with M = 1 - d, where W has +1
        on edges, -1 elsewhere, and 0 on the diagonal. This equals 4*cc_lp(d) up
        to an additive constant, i.e. it is exactly the CGW metric-LP objective.
        """
        edge_index, edge_weight = connectivity_to_edge_index(adj, edge_weight)

        if self.num_pivots is None:
            m = out.size(0)
            sub_out = out
            W = self._fresh_W(m, out.device)
            # assuming graph is undirected
            if edge_weight is None:
                W[edge_index[0], edge_index[1]] = 1.0
            else:
                W[edge_index[0], edge_index[1]] = edge_weight
        else:
            # pivot-based loss computation
            num_nodes = out.size(0)
            num_pivots = min(self.num_pivots, num_nodes)

            pivots = torch.randperm(num_nodes, device=out.device)[:num_pivots]
            nodes, pivot_edge_index, _, _ = tg_utils.k_hop_subgraph(
                pivots, 1, edge_index, relabel_nodes=True, num_nodes=num_nodes
            )
            sub_out = out[nodes]
            m = sub_out.size(0)
            W = self._fresh_W(m, out.device)
            W[pivot_edge_index[0], pivot_edge_index[1]] = 1.0  # assuming undirected

        dists = torch.cdist(sub_out, sub_out) / 2
        C = 1 - dists
        diff = torch.norm(W - C, p='fro')**2
        norm_C = torch.norm(C, p='fro')**2
        aux = diff - norm_C
        aux = aux / (m * m)  # normalize by num elements

        # Returned as a dict so tgp's get_loss_value() picks it up (a bare
        # tensor made it return None, which is what silently zeroed this loss).
        return {"cc_loss": AUX_LOSS_WEIGHT * aux}

    def forward(
        self,
        x: Tensor,
        adj: Adj,
        edge_weight: OptTensor = None,
        so: Optional[SelectOutput] = None,
        batch: Optional[Tensor] = None,
        lifting: bool = False,
        **kwargs,
    ) -> PoolingOutput:
        """SRCPooling-style forward.

        - Compute embeddings via single GCN/Linear
        - Build clusters from embeddings and input connectivity
        - Reduce graph to cluster-level nodes and edges
        - Return PoolingOutput with a loss dict
        """
        if lifting:
            # Lift
            if so is None:
                raise ValueError("SelectOutput (so) cannot be None for lifting")
            x_lifted = self.lift(x_pool=x, so=so)
            return x_lifted

        # Select
        so = self.select(x=x, adj=adj, edge_weight=edge_weight, batch=batch)
        loss = self.compute_loss(so.h, adj, edge_weight=edge_weight)

        # Reduce
        x_pooled, batch_pooled = self.reduce(x=x, so=so, batch=batch)

        # Connect
        edge_index_pooled, edge_weight_pooled = self.connect(
            edge_index=adj,
            so=so,
            edge_weight=edge_weight,
            batch_pooled=batch_pooled,
        )

        out = PoolingOutput(
            x=x_pooled,
            edge_index=edge_index_pooled,
            edge_weight=edge_weight_pooled,
            batch=batch_pooled,
            so=so,
            loss=loss,
        )
        return out

    @property
    def is_dense(self) -> bool:
        # CCPooler operates in sparse mode
        return False

    @staticmethod
    def data_transforms():
        # No extra transforms required
        return None

    def reset_parameters(self):
        self.selector.reset_parameters()
        self.reducer.reset_parameters()
        self.connector.reset_parameters()
        self.lifter.reset_parameters()

    def to(self, device):
        self.selector.to(device)
        self.reducer.to(device)
        self.connector.to(device)
        self.lifter.to(device)
        self.W = self.W.to(device) if self.W is not None else None
        return self
