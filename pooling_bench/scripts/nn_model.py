from math import ceil
import time
import torch
import torch.nn.functional as F
from torch_geometric.nn import GINConv, MLP, DenseGINConv, PANConv
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.nn import global_add_pool

from scripts.pooling.cc_pool import CCPooler

# TGP integration
from tgp.poolers import get_pooler, pooler_map


class GIN_Pool_Net(torch.nn.Module):
    """Model that uses poolers from the tgp library (plus custom CCPooler) as in graph_classification.py.

    Architecture:
    - GCNConv -> Pooler -> GCNConv (DenseGCNConv if pooler.is_dense)
    - Global pooling -> Linear -> LogSoftmax
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_channels=64,
                 activation='ELU',
                 average_nodes=100,
                 max_nodes=None,
                 pooling=None,
                 pooler = None,
                 pool_ratio=0.1,
                 k_order=None,
                 linear_cc=False,
                 num_layers_pre=2,
                 num_pivots=None,
                 k_sep=2,
                 ):
        super().__init__()
        self.pooling = pooling
        self.k_sep = k_sep

        self.conv_layers_pre = torch.nn.ModuleList()
        self.act = activation_resolver(activation)

        for _ in range(num_layers_pre):
            # First MP layer
            if pooling != 'pan':
                mlp = MLP([in_channels, hidden_channels, hidden_channels], act=activation)
                self.conv_layers_pre.append(GINConv(nn=mlp, train_eps=False))
            else:
                self.conv_layers_pre.append(PANConv(in_channels, hidden_channels, filter_size=2))
            in_channels = hidden_channels

            

        if pooler is not None:
            self.pooler = pooler
        else:
            pooled_nodes = ceil(pool_ratio * average_nodes)
            if pooling in ['diff', 'mincut', 'dmon', 'acc', 'bnpool', 'hosc', 'jb']:
                pooler_kwargs = {
                    "in_channels": hidden_channels,
                    "k": pooled_nodes,
                }
            elif pooling == 'kmis':
                pooler_kwargs = {
                    "in_channels": hidden_channels,
                    "order_k": k_order,
                }
            elif pooling in ['nmf', 'ndp']:
                pooler_kwargs = {
                    "k" : pooled_nodes,
                    "cached": True,
                }  
            else:
                pooler_kwargs = {
                    "in_channels": hidden_channels,
                    "ratio": pool_ratio,
                }
            
            if pooling == "ccpool":
                # Use custom CCPooler implemented to be SRCPooling-compatible
                self.pooler = CCPooler(in_channels=hidden_channels,
                                    hidden_channels=hidden_channels,
                                    ratio=pool_ratio,
                                    max_nodes=max_nodes,
                                    linear=linear_cc,
                                    num_pivots=num_pivots)
            else:
                # Fallback to TGP poolers
                self.pooler = get_pooler(pooling, **pooler_kwargs)

        # Second MP layer
        mlp = MLP([hidden_channels, hidden_channels, hidden_channels], act=activation, norm=None)
        if getattr(self.pooler, "is_dense", False):
            self.conv_post = DenseGINConv(nn=mlp, train_eps=False)
        else:
            self.conv_post = GINConv(nn=mlp, train_eps=False)
            

        # Readout layer
        self.mlp = MLP([hidden_channels, hidden_channels, hidden_channels//2, out_channels], 
                        act=activation,
                        norm=None,
                        dropout=0.5)

    def forward(self, data, verbose: bool = False):
        x, edge_index, edge_weight, batch = data.x, data.edge_index, getattr(data, "edge_weight", None), data.batch

        num_nodes = x.size(0)
        # Convert to SparseTensor adjacency for GCNConv compatibility
        #adj = connectivity_to_sparse_tensor(edge_index, edge_weight, num_nodes)
        adj = edge_index

        # First MP layer
        if self.pooling != 'pan':
            for conv in self.conv_layers_pre:
                x = conv(x, adj)
                x = self.act(x)
        else: 
            M = adj
            for conv in self.conv_layers_pre:
                x, M = conv(x, M)
                x = self.act(x)
                

        # Preprocessing and pooling
        if self.pooling == 'pan':
            out = self.pooler(x=x, adj=M, edge_weight=None, batch=batch)
        else:
            x, adj, mask = self.pooler.preprocessing(x=x, edge_index=adj, edge_weight=edge_weight, batch=batch)
            out = self.pooler(x=x, adj=adj, edge_weight=edge_weight, batch=batch, mask=mask)
            if self.pooling == 'sep':
                for _ in range(self.k_sep-1):
                    out = self.pooler(x=out.x, adj=out.edge_index, edge_weight=out.edge_weight, batch=out.batch, mask=out.mask)

        x_pool, adj_pool = out.x, out.edge_index
        if verbose:
            print(f"Pooled from {num_nodes} to {x_pool.size(0)} nodes, {x_pool.size(0)/num_nodes:.2f} ratio.") 

        # Second MP layer
        x = self.conv_post(x_pool, adj_pool)
        x = self.act(x)

        # Global pooling
        #if self.pooling == 'sep':
            # perform global pooling using torch_geometric.nn.global_add_pool
        x = global_add_pool(x, out.batch)
        #else:
           # x = self.pooler.global_pool(x, reduce_op="sum", batch=out.batch)

        # Readout
        x = self.mlp(x)

        aux_loss = torch.tensor(0.0, device=x_pool.device)
        aux_val = None
        if getattr(out, "loss", None) is not None:
            # Two separate faults lived here.
            #
            # (1) CCPooler builds PoolingOutput(..., loss=<bare tensor>), but
            #     tgp's get_loss_value() expects a dict of named losses and
            #     returns None for a bare tensor.  torch.tensor(None) then
            #     raised TypeError, the bare `except Exception: pass` swallowed
            #     it, and aux_loss stayed at exactly 0.0.  The CC objective was
            #     never retrieved at all -- which is why the selector's GCN
            #     never received a gradient and stayed at its random init.
            #     Fall back to out.loss when get_loss_value() gives nothing.
            #
            # (2) Even when a value did come back, `torch.tensor(aux_val, ...)`
            #     copies the data and DROPS THE AUTOGRAD HISTORY.  Only the
            #     `isinstance(aux_val, list)` branch preserved the gradient.
            #
            # The bare except also hid both; it now warns.
            try:
                aux_val = out.get_loss_value()
            except Exception as e:
                import warnings
                warnings.warn(f'get_loss_value() failed ({type(e).__name__}: {e}); '
                              'falling back to out.loss')
                aux_val = None

            if aux_val is None:
                aux_val = getattr(out, "loss", None)

            if isinstance(aux_val, (list, tuple)):
                aux_loss = sum(aux_val)
            elif isinstance(aux_val, dict):
                aux_loss = sum(aux_val.values())
            elif torch.is_tensor(aux_val):
                aux_loss = aux_val.to(x_pool.device)
            elif aux_val is not None:
                aux_loss = torch.as_tensor(aux_val, device=x_pool.device,
                                           dtype=torch.float32)

            if torch.is_tensor(aux_loss) and not aux_loss.requires_grad \
                    and torch.is_grad_enabled() and self.training:
                import warnings
                warnings.warn(
                    f'[{self.pooling}] auxiliary loss is detached from the graph; '
                    'it cannot train the pooler.')


        return F.log_softmax(x, dim=-1), aux_loss
    
    def to(self, device):
        self.conv_layers_pre.to(device)
        self.conv_post.to(device)
        self.mlp.to(device)
        self.pooler.to(device)
        return super().to(device)