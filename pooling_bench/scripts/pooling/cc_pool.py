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



class CCSelector(Select):
    def __init__(self, in_channels: int, hidden_channels: int, ratio: float, s_inv_op: SinvType = "transpose", linear: bool = False) -> None:
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
        # Embeddings
        if self.linear:
            h = self.model(x, edge_weight=edge_weight)
        else:
            h = self.model(x, edge_index, edge_weight=edge_weight)
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
        Optimized connected-component style clustering using precomputed distances and GPU acceleration.
        - Precomputes all pairwise distances once on the GPU.
        - Assumes a complete graph for traversal.
        """
        num_nodes = out.size(0)
        undirected_edge_index = edge_index[:, edge_index[0] < edge_index[1]]
        device = out.device

        # Precompute pairwise distances on GPU
        out_half = out
        row_out = out_half[undirected_edge_index[0]]
        col_out = out_half[undirected_edge_index[1]]
        dists_sq = torch.sum((row_out - col_out) ** 2, dim=1)

        k = max(int((1-self.pool_ratio) * dists_sq.size(0)), 1)
        topk_vals, topk_indices = torch.topk(dists_sq, k, largest=False)
        filtered_edges = undirected_edge_index[:, topk_indices]

        filtered_edges = torch.cat([filtered_edges, filtered_edges[[1,0],:]], dim=1)  # make directed
        
        sp_graph = tg_utils.to_scipy_sparse_matrix(filtered_edges, num_nodes=num_nodes)
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
    Learns a single GCN layer using the loss from `train_lpmodel` in `src/train.py`,
    then forms clusters by applying `make_cc_clusters` (also from `src/train.py`) on
    the learned node embeddings using a provided distance threshold.

    The pooling reduces the input graph to cluster-level nodes by aggregating
    features per cluster (default: sum) and collapsing edges between clusters
    (default: sum), removing self-loops.

    Args:
        in_channels (int): Input feature dimension per node.
        hidden_channels (int): Dimension of the learned embedding (GCN output).
        threshold (float): Distance threshold for `make_cc_clusters`.
        device (str or torch.device): Training/evaluation device. Defaults to CUDA if available.
        epochs (int): Max training epochs for the single-layer GCN.
        lr (float): Learning rate for Adam.
        weight_decay (float): Weight decay for Adam.
        patience (int): Early stopping patience (uses moving average like `train_lpmodel`).
        random_pivots (Optional[int]): Pivot subsampling used by `train_lpmodel` to scale training.
        aggr_x (str): Feature aggregation over clusters (e.g., 'sum', 'mean', 'max').
        aggr_edge (str): Edge aggregation between clusters for SparseTensor coalesce.
        remove_self_loops (bool): Remove self-loops in the reduced graph.

    Notes:
        - This module trains on the provided graph the first time `forward` is called
          (or when `fit` is invoked explicitly). Subsequent calls reuse the trained
          model for the same graph.
        - It expects a single graph (no batched multi-graph input). If `batch` is
          provided, it will be collapsed to a single graph with a single batch label.
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
            self.W = torch.ones((max_nodes, max_nodes))*(-1.0)
            self.W.fill_diagonal_(0.0)


    def compute_loss(self, out: Tensor, adj: Adj, edge_weight: OptTensor = None) -> Tensor:
        """Compute the train_lpmodel-style loss on current embeddings.
        Loss per graph: ||W - (1 - cdist/2)||_F^2 - ||(1 - cdist/2)||_F^2
        where W has +1 on edges, -1 elsewhere, and 0 on diagonal.
        """

        edge_index, edge_weight = connectivity_to_edge_index(adj, edge_weight)
        
        if self.num_pivots is None:
            m = out.size(0)
            if self.W is None:
                W = torch.ones((m, m), device=out.device) * (-1.0)
                W.fill_diagonal_(0.0)
                # assuming graph is undirected
                if edge_weight is None:
                    W[edge_index[0], edge_index[1]] = 1.0 
                else:
                    W[edge_index[0], edge_index[1]] = edge_weight
            else:
                W = self.W[:m, :m]
                # assuming graph is undirected
                if edge_weight is None:
                    W[edge_index[0], edge_index[1]] = 1.0 
                else:
                    W[edge_index[0], edge_index[1]] = edge_weight
            dists = torch.cdist(out, out) / 2
        else:
            # pivot-based loss computation
            num_nodes = out.size(0)
            num_pivots = min(self.num_pivots, num_nodes)
            
            pivots = torch.randperm(out.size(0))[:num_pivots]
            nodes, pivot_edge_index, _, _ = tg_utils.k_hop_subgraph(pivots, 1, edge_index, relabel_nodes=True, num_nodes=out.size(0))
            sub_out = out[nodes]
            m = sub_out.size(0)
            #print('sub_nodes', m)
            if self.W is None:
                W = torch.ones((m, m), device=out.device) * (-1.0)
                W.fill_diagonal_(0.0)
                W[pivot_edge_index[0], pivot_edge_index[1]] = 1.0 # assuming graph is undirected
            else:
                W = self.W[:m, :m]
                W[pivot_edge_index[0], pivot_edge_index[1]] = 1.0 # assuming graph is undirected
            dists = torch.cdist(sub_out, sub_out) / 2

        C = 1 - dists
        diff = torch.norm(W - C, p='fro')**2
        norm_C = torch.norm(C, p='fro')**2
        aux = diff - norm_C
        aux /= (m * m)  # normalize by num elements

        return aux

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
