import os
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['NUMEXPR_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'

import torch
import torch.nn.functional as F
import torch_geometric as tg
import pandas as pd
import numpy as np
import time

from torch_geometric import seed_everything
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.nn import Node2Vec
from tqdm import tqdm

from load_utils import EXPWL1Dataset
from models import GNNModel
from utils import compute_cost_from_clustering_complete_graph
from train import train_linkmodel_batch, make_cc_clusters

def add_ones(data):
    data.x = torch.ones((data.num_nodes, 1), dtype=torch.float32)
    return data

def compute_node2vec_features_batch(dataset, n2v_dim=128, n2v_epochs=100,
                                     n2v_walk_length=10, n2v_context_size=10,
                                     n2v_lr=0.01, loader_batch_size=64):
    """
    Compute Node2Vec embeddings for every graph in *dataset*.

    Trains a **single** Node2Vec model per DataLoader batch.  Because PyG
    batches multiple graphs into one disconnected graph (no cross-graph
    edges), random walks naturally stay within individual graphs, so this
    is semantically equivalent to per-graph training but much faster.

    After training, the learned node embeddings are written into
    ``batch.x`` and ``to_data_list()`` splits them back per graph.

    Returns a plain Python list of ``Data`` objects.
    """
    loader = DataLoader(dataset, batch_size=loader_batch_size, shuffle=False)
    processed: list = []
    clean_count = 0
    for batch in tqdm(loader, desc='Computing Node2Vec features (batched)'):
        batch.to(device)
        if batch.edge_index.size(1) == 0:
            # Entire batch has no edges — fall back to random features
            batch.x = torch.randn(batch.num_nodes, n2v_dim)
        else:
            n2v = Node2Vec(
                batch.edge_index,
                embedding_dim=n2v_dim,
                walk_length=n2v_walk_length,
                context_size=n2v_context_size,
                num_nodes=batch.num_nodes,
            ).to(device)
            n2v_loader = n2v.loader(batch_size=2048, shuffle=True, num_workers=0)
            optimizer = torch.optim.Adam(n2v.parameters(), lr=n2v_lr)

            n2v.train()
            for _ in range(n2v_epochs):
                for pos_rw, neg_rw in n2v_loader:
                    pos_rw, neg_rw = pos_rw.to(device), neg_rw.to(device)
                    optimizer.zero_grad()
                    loss = n2v.loss(pos_rw, neg_rw)
                    loss.backward()
                    optimizer.step()

            n2v.eval()
            with torch.no_grad():
                batch.x = n2v().detach()
            
            del n2v, n2v_loader, optimizer
            

        clean_count += 1
        if clean_count % 10 == 0:
            torch.cuda.empty_cache()

            

        processed.extend(batch.to_data_list())
    return processed

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
timestamp = time.strftime("%Y%m%d-%H%M%S")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

datasets       = ['REDDIT-BINARY']
hidden_dims    = [16, 64, 256]          # sweep over hidden_channels
thresholds     = np.arange(0.0, 1.05, 0.05)      # same grid used in main_inductive3

n_iters        = 5
patience       = 10
num_epochs     = 1000
batch_size     = 64
lr             = 0.01
weight_decay   = 0.0

results = []

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
for dataset_name in datasets:

    # --- load dataset -------------------------------------------------------
    if dataset_name in ['MUTAG', 'NCI1', 'REDDIT-BINARY']:
        dataset = TUDataset(root='../data/TUDataset', name=dataset_name,
                            pre_transform=None, use_node_attr=False, force_reload=False, transform=add_ones)
    elif dataset_name == 'ogbg-molhiv':
        dataset = PygGraphPropPredDataset(name='ogbg-molhiv', root='../data/OGB')
        dataset.data.x = dataset.data.x.to(torch.float32)
        dataset.data.y = dataset.data.y.view(-1)
    else:
        dataset = EXPWL1Dataset(root=f'../data/{dataset_name}')
    in_channels = 128
    
    print(f'\n=== Dataset: {dataset_name}  in_channels={in_channels} ===')

    for hidden_channels in hidden_dims:
        out_channels = hidden_channels          # embedding dim = hidden_channels for LinkGNN

        for i in range(n_iters):
            seed_everything(i * 1000 + 6199)

            # --- data split -------------------------------------------------
            rnd_idx      = np.random.permutation(len(dataset))
            split_idx    = int(0.8 * len(dataset))
            train_dataset = dataset[rnd_idx[:split_idx]]
            val_dataset   = dataset[rnd_idx[split_idx:int(0.9 * len(dataset))]]
            test_dataset  = dataset[rnd_idx[int(0.9 * len(dataset)):]]
            
            train_dataset = compute_node2vec_features_batch(train_dataset, n2v_dim=in_channels)
            val_dataset = compute_node2vec_features_batch(val_dataset, n2v_dim=in_channels)
            test_dataset = compute_node2vec_features_batch(test_dataset, n2v_dim=in_channels)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)
            test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

            # --- build model ------------------------------------------------
            model = GNNModel(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                out_channels=out_channels,
                bias=False,
                input_linearity=False,
            ).to(device)
            print(model.convs[0].lin.weight.shape)

            # --- train ------------------------------------------------------
            seed_everything(6199 + i * 1000)

            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device)
            train_time_start = time.time()

            model = train_linkmodel_batch(
                model,
                train_loader,
                val_loader,
                device,
                epochs=num_epochs,
                lr=lr,
                wd=weight_decay,
                patience=patience,
                batch_size=batch_size,
            )

            train_time = time.time() - train_time_start
            gpu_peak_memory_mb = (
                torch.cuda.max_memory_allocated(device) / 1024 ** 2
                if device.type == 'cuda' else None
            )

            print(f"  hidden={hidden_channels}, iter={i}: trained in {train_time:.1f}s"
                  + (f", GPU peak {gpu_peak_memory_mb:.0f} MB" if gpu_peak_memory_mb is not None else ""))

            # --- pre-compute model outputs on test set (avoid redundant forward passes) --
            # We run inference once per threshold sweep to avoid re-running
            # the GNN for every threshold. Outputs and edge_indices are cached.
            model.eval()
            test_outs       = []   # list of (out, edge_index) per batch
            with torch.no_grad():
                for data in test_loader:
                    data.to(device)
                    out = model(data)
                    out = F.normalize(out, p=2, dim=1)
                    test_outs.append((out.detach(), data.edge_index.detach(), data.num_nodes, data.num_edges))

            # --- threshold sweep on test set --------------------------------
            for threshold in thresholds:
                cost           = 0.0
                inference_time = 0.0

                for out, edge_index, num_nodes, num_edges in test_outs:
                    time_start = time.time()
                    clustering = make_cc_clusters(out, edge_index, threshold, device)
                    time_end   = time.time()

                    cost           += compute_cost_from_clustering_complete_graph(
                                        clustering.to(device), edge_index)
                    inference_time += time_end - time_start

                row = {
                    'name':               dataset_name,
                    'method':             'LinkGNN',
                    'hidden_channels':    hidden_channels,
                    'in_channels':        in_channels,
                    'iter':               i,
                    'threshold':          round(float(threshold), 4),
                    'lr':                 lr,
                    'weight_decay':       weight_decay,
                    'patience':           patience,
                    'train_time':         train_time,
                    'gpu_peak_memory_mb': gpu_peak_memory_mb,
                    'inference_time':     inference_time,
                    'test_cost':          cost,
                }
                results.append(row)

            # save after each (dataset, hidden_channels, iter) triple
            pd.DataFrame(results).to_csv(
                f'../results/linkgnn_threshold_sweep_{timestamp}.csv', index=False
            )
            torch.cuda.empty_cache()

print("\nDone. Results saved to ../results/linkgnn_threshold_sweep_{}.csv".format(timestamp))
