"""
Ablation study for LinkGNN on REDDIT-BINARY.

Five independent ablations (each compared against a full-pipeline baseline):
  1) No pivot sampling in training  (use the full graph per batch, no k-hop subgraph)
  2) No embedding normalisation in training or testing
  3) Calibrated CGW-complete rounding instead of the percentile sweep at test time
  4) Calibrated pivot rounding instead of the percentile sweep at test time
  5) Laplacian positional encodings as node features (instead of all-ones)
"""

import os
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['NUMEXPR_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'

import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import time

from torch_geometric import seed_everything
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_scipy_sparse_matrix
import scipy.sparse as sp
from torch_geometric.nn import Node2Vec
from tqdm import tqdm

from models import GNNModel
from utils import compute_cost_from_clustering_complete_graph
from train import make_cc_clusters, make_pivot_clusters
# ---------------------------------------------------------------------------
# Rounding helpers, used only by the charikar_complete / pivot_rounding arms.
#
# Those arms applied the absolute Charikar-Guruswami-Wirth constants (ball
# radius 1/2, mean radius 1/4) straight to the learned distances.  Those
# constants assume the LP scale, but the training loss only fixes the *ordering*
# of the distances, never their magnitude -- empirically the intra/inter
# boundary sits near d ~ 0.05.  Every candidate therefore passed both tests and
# the first pivot absorbed its whole connected component, so the arms measured a
# degenerate clustering rather than a design choice.  We rescale first.
#
# min(1, s*d) is still a metric for any s > 0, so rescaling does not invalidate
# anything the rounding relies on.
# ---------------------------------------------------------------------------


def _undirected(edge_index):
    return edge_index[:, edge_index[0] < edge_index[1]]


def _components(filtered_edges, num_nodes):
    if filtered_edges.numel() == 0:
        return torch.arange(num_nodes, dtype=torch.long)
    filtered_edges = torch.cat([filtered_edges, filtered_edges[[1, 0], :]], dim=1)
    g = to_scipy_sparse_matrix(filtered_edges.cpu(), num_nodes=num_nodes)
    _, labels = sp.csgraph.connected_components(g, directed=False, return_labels=True)
    return torch.from_numpy(labels).long()


def split_disconnected(labels, edge_index, num_nodes):
    """Split each cluster into its connected components in G.

    Strictly lowers the CC cost (removes sum_{a<b} |S_a||S_b| missing-edge
    penalties, adds no cut edge), so it is always safe.
    """
    labels = labels.cpu()
    ei = _undirected(edge_index).cpu()
    return _components(ei[:, labels[ei[0]] == labels[ei[1]]], num_nodes)


def _scale_grid(d_edges, num_targets=16, lo=0.02, hi=2.0):
    """Scales anchored on the observed edge-distance median.

    A fixed absolute grid does not work: the scale the embedding lands on varies
    by orders of magnitude between runs, and a fixed grid silently saturates at
    its endpoint.
    """
    ref = float(torch.median(d_edges).item()) if d_edges.numel() else 1.0
    if not np.isfinite(ref) or ref <= 0:
        return np.array([1.0])
    return np.unique(np.concatenate([np.geomspace(lo, hi, num_targets) / ref, [1.0]]))


def round_cgw_complete(out, edge_index, num_nodes, scale=1.0, seed=6199):
    """CGW pivot rounding for complete +-1 graphs, on rescaled distances."""
    n = int(num_nodes)
    neighbours = [[] for _ in range(n)]
    src, dst = edge_index.cpu().tolist()
    for u, v in zip(src, dst):
        neighbours[u].append(v)

    rng = np.random.default_rng(seed)
    unclustered = set(range(n))
    labels = torch.full((n,), -1, dtype=torch.long)
    k = 0
    while unclustered:
        pivot = int(rng.choice(list(unclustered)))
        unclustered.discard(pivot)
        cand = [v for v in neighbours[pivot] if v in unclustered]
        if cand:
            cand_t = torch.tensor(cand, dtype=torch.long, device=out.device)
            d = torch.norm(out[cand_t] - out[pivot].unsqueeze(0), p=2, dim=1) / 2.0
            d = torch.clamp(d * scale, max=1.0)
            in_ball = d <= 0.5
            if bool(in_ball.any()) and d[in_ball].mean().item() <= 0.25:
                ball = cand_t[in_ball].cpu()
                labels[ball] = k
                unclustered.difference_update(ball.tolist())
        labels[pivot] = k
        k += 1
    return labels


def calibrate_scale_by_cost(out, edge_index, num_nodes, cost_fn, seed=6199,
                            rounding=None):
    """Pick the rescaling by the true CC cost of the clustering it produces.

    Selecting on the LP value instead over-separates: the LP sums ~n^2/2
    non-edge terms against only |E| edge terms, so its minimiser fragments the
    clustering.  Returns (best_scale, best_labels, best_cost).
    """
    ei = _undirected(edge_index.to(out.device))
    d_e = (torch.norm(out[ei[0]] - out[ei[1]], p=2, dim=1) / 2.0
           if ei.size(1) else torch.ones(1, device=out.device))
    rounding = rounding or round_cgw_complete
    best = (1.0, None, float('inf'))
    for s_ in _scale_grid(d_e):
        labels = split_disconnected(
            rounding(out, edge_index, num_nodes, scale=float(s_), seed=seed),
            edge_index, num_nodes)
        c = cost_fn(labels)
        if c < best[2]:
            best = (float(s_), labels, c)
    return best


# ---------------------------------------------------------------------------
# Dataset helper
# ---------------------------------------------------------------------------
def add_ones(data):
    data.x = torch.ones((data.num_nodes, 1), dtype=torch.float32)
    return data


def compute_node2vec_features_batch(dataset, n2v_dim=128, n2v_epochs=100,
                                     n2v_walk_length=10, n2v_context_size=10,
                                     n2v_lr=0.01, loader_batch_size=64, device='cpu'):
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
                   


# ===========================================================================
# Modified training / inference routines for each ablation
# ===========================================================================

# ---- Baseline: exact copy of train_linkmodel_batch (for clarity) ----------
def train_linkmodel_batch_baseline(model, train_loader, val_loader, device,
                                   epochs=100, lr=0.01, wd=0.0,
                                   patience=10, batch_size=64):
    """Standard LinkGNN training with pivot sampling & normalisation."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    best_loss = float('inf')
    best_model = model.state_dict()
    count = 0

    for epoch in (pbar := tqdm(range(epochs), desc="[baseline] Training")):
        model.train()
        for data in train_loader:
            m = data.num_nodes
            edge_index = data.edge_index

            pos_edge_mask = getattr(data, 'edge_weight', None)
            if pos_edge_mask is not None:
                pos_edge_mask = pos_edge_mask > 0
                edge_index = edge_index[:, pos_edge_mask]

            weight_matrix = getattr(data, 'W', None)
            optimizer.zero_grad()

            if batch_size is None:
                subgraph_data = data
                subgraph_data.to(device)
                num_subgraph_nodes = m
                subgraph_edge_index = edge_index.to(device)
                subgraph_nodes = torch.arange(m)
            else:
                pivots = torch.randperm(m)[:batch_size].tolist()
                subgraph_nodes, subgraph_edge_index, _, _ = k_hop_subgraph(
                    pivots, 1, edge_index, relabel_nodes=True, num_nodes=m)
                num_subgraph_nodes = subgraph_nodes.size(0)

                subgraph_data = Data()
                subgraph_data.x = data.x[subgraph_nodes.cpu()].to(device)
                subgraph_data.edge_index = subgraph_edge_index.to(device)

            out = model(subgraph_data)
            out = F.normalize(out, p=2, dim=1)

            dists = torch.cdist(out, out) / 2
            co_cluster_matrix = 1 - dists

            if weight_matrix is not None:
                curr_W = weight_matrix[subgraph_nodes, :][:, subgraph_nodes].to(device)
            else:
                curr_W = torch.ones((num_subgraph_nodes, num_subgraph_nodes),
                                    device=out.device) * (-1.0)
                curr_W.fill_diagonal_(0.0)
                curr_W[subgraph_edge_index[0], subgraph_edge_index[1]] = 1.0
                curr_W[subgraph_edge_index[1], subgraph_edge_index[0]] = 1.0

            diff = torch.norm(curr_W - co_cluster_matrix, p='fro') ** 2
            norm = torch.norm(co_cluster_matrix, p='fro') ** 2
            loss = diff - norm

            loss.backward()
            optimizer.step()

            del subgraph_data, subgraph_nodes, subgraph_edge_index

        del out, loss, diff, norm, dists, curr_W
        if epoch % 10 == 0:
            torch.cuda.empty_cache()

        # --- validation ---
        tot_loss = 0.0
        model.eval()
        with torch.no_grad():
            for val_data in val_loader:
                val_data.to(device)
                m = val_data.num_nodes
                val_edge_index = val_data.edge_index

                val_out = model(val_data)
                val_out = F.normalize(val_out, p=2, dim=1)

                dists = torch.cdist(val_out, val_out) / 2
                co_cluster_matrix = 1 - dists

                val_weight_matrix = getattr(val_data, 'W', None)
                if val_weight_matrix is not None:
                    curr_W = val_weight_matrix.to(device)
                else:
                    curr_W = torch.ones((m, m), device=val_out.device) * (-1.0)
                    curr_W.fill_diagonal_(0.0)
                    curr_W[val_edge_index[0], val_edge_index[1]] = 1.0
                    curr_W[val_edge_index[1], val_edge_index[0]] = 1.0

                diff = torch.norm(curr_W - co_cluster_matrix, p='fro') ** 2
                norm = torch.norm(co_cluster_matrix, p='fro') ** 2
                val_loss = diff - norm
                tot_loss += val_loss.item()

        if tot_loss <= best_loss - 1:
            best_loss = tot_loss
            best_model = model.state_dict()
            count = 0
        elif epoch >= patience:
            count += 1
            if count == patience:
                break

    model.load_state_dict(best_model)
    model.eval()
    return model


# ---- Ablation 1: NO pivot sampling (use full mini-batch, no sub-sampling) -
def train_linkmodel_batch_no_pivot(model, train_loader, val_loader, device,
                                   epochs=100, lr=0.01, wd=0.0,
                                   patience=10):
    """Training without pivot/subgraph sampling — every graph in the batch is
    used whole (batch_size=None path)."""
    return train_linkmodel_batch_baseline(
        model, train_loader, val_loader, device,
        epochs=epochs, lr=lr, wd=wd, patience=patience,
        batch_size=None,           # <--- the ablation: no pivot subsampling
    )


# ---- Ablation 2: NO normalisation ----------------------------------------
def train_linkmodel_batch_no_norm(model, train_loader, val_loader, device,
                                  epochs=100, lr=0.01, wd=0.0,
                                  patience=10, batch_size=64):
    """Training without L2-normalising embeddings."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    best_loss = float('inf')
    best_model = model.state_dict()
    count = 0

    for epoch in (pbar := tqdm(range(epochs), desc="[no-norm] Training")):
        model.train()
        for data in train_loader:
            m = data.num_nodes
            edge_index = data.edge_index

            pos_edge_mask = getattr(data, 'edge_weight', None)
            if pos_edge_mask is not None:
                pos_edge_mask = pos_edge_mask > 0
                edge_index = edge_index[:, pos_edge_mask]

            weight_matrix = getattr(data, 'W', None)
            optimizer.zero_grad()

            pivots = torch.randperm(m)[:batch_size].tolist()
            subgraph_nodes, subgraph_edge_index, _, _ = k_hop_subgraph(
                pivots, 1, edge_index, relabel_nodes=True, num_nodes=m)
            num_subgraph_nodes = subgraph_nodes.size(0)

            subgraph_data = Data()
            subgraph_data.x = data.x[subgraph_nodes.cpu()].to(device)
            subgraph_data.edge_index = subgraph_edge_index.to(device)

            out = model(subgraph_data)
            # >>> ABLATION: skip normalisation <<<

            dists = torch.cdist(out, out) / 2
            co_cluster_matrix = 1 - dists

            if weight_matrix is not None:
                curr_W = weight_matrix[subgraph_nodes, :][:, subgraph_nodes].to(device)
            else:
                curr_W = torch.ones((num_subgraph_nodes, num_subgraph_nodes),
                                    device=out.device) * (-1.0)
                curr_W.fill_diagonal_(0.0)
                curr_W[subgraph_edge_index[0], subgraph_edge_index[1]] = 1.0
                curr_W[subgraph_edge_index[1], subgraph_edge_index[0]] = 1.0

            diff = torch.norm(curr_W - co_cluster_matrix, p='fro') ** 2
            norm = torch.norm(co_cluster_matrix, p='fro') ** 2
            loss = diff - norm

            loss.backward()
            optimizer.step()

            del subgraph_data, subgraph_nodes, subgraph_edge_index

        del out, loss, diff, norm, dists, curr_W
        if epoch % 10 == 0:
            torch.cuda.empty_cache()

        # --- validation (also without normalisation) ---
        tot_loss = 0.0
        model.eval()
        with torch.no_grad():
            for val_data in val_loader:
                val_data.to(device)
                m = val_data.num_nodes
                val_edge_index = val_data.edge_index

                val_out = model(val_data)
                # >>> ABLATION: skip normalisation <<<

                dists = torch.cdist(val_out, val_out) / 2
                co_cluster_matrix = 1 - dists

                val_weight_matrix = getattr(val_data, 'W', None)
                if val_weight_matrix is not None:
                    curr_W = val_weight_matrix.to(device)
                else:
                    curr_W = torch.ones((m, m), device=val_out.device) * (-1.0)
                    curr_W.fill_diagonal_(0.0)
                    curr_W[val_edge_index[0], val_edge_index[1]] = 1.0
                    curr_W[val_edge_index[1], val_edge_index[0]] = 1.0

                diff = torch.norm(curr_W - co_cluster_matrix, p='fro') ** 2
                norm = torch.norm(co_cluster_matrix, p='fro') ** 2
                val_loss = diff - norm
                tot_loss += val_loss.item()

        if tot_loss <= best_loss - 1:
            best_loss = tot_loss
            best_model = model.state_dict()
            count = 0
        elif epoch >= patience:
            count += 1
            if count == patience:
                break

    model.load_state_dict(best_model)
    model.eval()
    return model


def make_cc_clusters_no_norm(out, edge_index, threshold, device):
    """Deprecated: kept only so old result files remain reproducible.
    The truncated metric is now the default, so this is `make_cc_clusters`."""
    return make_cc_clusters(out, edge_index, threshold, device)


# ---- Ablation 3 helper: CGW-complete rounding, rescaled --------------------
def make_charikarcomplete_clusters_batch(out, edge_index, device):
    """CGW pivot rounding for complete +-1 graphs, with the scale calibrated
    by true CC cost (see the note at the top of this file)."""
    o = F.normalize(out, p=2, dim=1)
    cost_fn = lambda lab: compute_cost_from_clustering_complete_graph(lab, edge_index)
    _, labels, _ = calibrate_scale_by_cost(o, edge_index, o.size(0), cost_fn)
    return labels


# ---- Ablation 4 helper: pivot rounding, rescaled ---------------------------
def make_pivot_clusters_batch(out, edge_index, device):
    """Pivot rounding at ball radius 1/2, on rescaled distances."""
    o = F.normalize(out, p=2, dim=1)
    cost_fn = lambda lab: compute_cost_from_clustering_complete_graph(lab, edge_index)

    def _pivot(emb, ei, num_nodes, scale=1.0, seed=6199):
        return make_pivot_clusters(emb * scale, device, threshold=0.5, seed=seed)

    _, labels, _ = calibrate_scale_by_cost(o, edge_index, o.size(0), cost_fn,
                                           rounding=_pivot)
    return labels


# ===========================================================================
# Main experiment loop
# ===========================================================================
def main():
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    # ------------ Hyperparameters ------------------------------------------
    dataset_name = 'REDDIT-BINARY'  # TUDataset name
    n_iters = 5
    patience = 500
    num_epochs = 5000
    batch_size = 64
    num_layers = 1
    out_channels = 64
    lr = 0.0001
    weight_decay = 0

    # ------------ Load datasets --------------------------------------------
    dataset = TUDataset(root='../data/TUDataset', name=dataset_name,
                        pre_transform=None, use_node_attr=False,
                        force_reload=False, transform=add_ones)
    in_channels = dataset[0].x.size(1)

    in_channels_n2v = 128
    

    print(f'Loaded dataset: {dataset_name} ({len(dataset)} graphs)')

    # ------------ Define ablation configs ----------------------------------
    ablations = {
        'node2vec': {
            'description': 'Node2Vec as node features',
        },
        'allones': {
            'description': 'Full pipeline (pivot sampling + norm + threshold)',
        },
        'no_pivot_sampling': {
            'description': 'No pivot/subgraph sampling during training',
        },
        'no_normalisation': {
            'description': 'No L2-normalisation of embeddings (in training and testing)',
        },
        'charikar_complete': {
            'description': 'Calibrated CGW-complete rounding (no percentile sweep)',
        },
        'pivot_rounding': {
            'description': 'Calibrated pivot rounding (no percentile sweep)',
        },
        
    }

    results = []

    for ablation_name, ablation_cfg in ablations.items():
        print(f'\n{"=" * 60}')
        print(f'  Ablation: {ablation_name}')
        print(f'  {ablation_cfg["description"]}')
        print(f'{"=" * 60}')

        for i in range(n_iters):
            seed_everything(6199 + i * 1000)

            # --- choose dataset / in_channels for this ablation ---
            if ablation_name == 'allones':
                curr_in_channels = in_channels
            else:
                curr_in_channels = in_channels_n2v


            print(curr_in_channels, out_channels)
            # --- fresh model each iteration ---
            model = GNNModel(
                in_channels=curr_in_channels,
                hidden_channels=out_channels,
                out_channels=out_channels,
                num_layers=num_layers,
                bias=False,
                input_linearity=False,
            ).to(device)

            # --- random train / val / test split ---
            rnd_idx = np.random.permutation(len(dataset))
            split_80 = int(0.8 * len(dataset))
            split_90 = int(0.9 * len(dataset))

            train_dataset = dataset[rnd_idx[:split_80]]
            val_dataset = dataset[rnd_idx[split_80:split_90]]
            test_dataset = dataset[rnd_idx[split_90:]]
            if ablation_name != 'allones':
                # Precompute Node2Vec features for this ablation
                train_dataset = compute_node2vec_features_batch(train_dataset, n2v_dim=in_channels_n2v, device=device)
                val_dataset = compute_node2vec_features_batch(val_dataset, n2v_dim=in_channels_n2v, device=device)
                test_dataset = compute_node2vec_features_batch(test_dataset, n2v_dim=in_channels_n2v, device=device)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            if ablation_name in ['charikar_complete', 'pivot_rounding']:
                test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
            else:
                test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            # ===== TRAINING =====
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device)
            train_start = time.time()

            if ablation_name == 'no_pivot_sampling':
                model = train_linkmodel_batch_no_pivot(
                    model, train_loader, val_loader, device,
                    epochs=num_epochs, lr=lr, wd=weight_decay,
                    patience=patience,
                )
            elif ablation_name == 'no_normalisation':
                # Ablation 2: no L2-normalisation, in training and at test time.
                # NOTE: without the normalisation the loss is unbounded below
                # (it is linear in the distances and the non-edge term rewards
                # spreading the embedding out), so it decreases monotonically
                # and the early-stopping test never fires. That is why this arm
                # trains several times longer for essentially the same cost.
                model = train_linkmodel_batch_no_norm(
                    model, train_loader, val_loader, device,
                    epochs=num_epochs, lr=lr, wd=weight_decay,
                    patience=patience, batch_size=batch_size,
                )
            else:
                # baseline, charikar_complete, pivot_rounding,
                # share the same training procedure
                model = train_linkmodel_batch_baseline(
                    model, train_loader, val_loader, device,
                    epochs=num_epochs, lr=lr, wd=weight_decay,
                    patience=patience, batch_size=batch_size,
                )

            train_time = time.time() - train_start
            gpu_peak_memory_mb = (
                torch.cuda.max_memory_allocated(device) / 1024 ** 2
                if device.type == 'cuda' else None
            )

            # ===== INFERENCE / CLUSTERING =====
            if ablation_name in ('charikar_complete', 'pivot_rounding'):
                # --- Ablation 3/4: rounding-based cluster extraction ---
                rounding_fn = (make_charikarcomplete_clusters_batch
                               if ablation_name == 'charikar_complete'
                               else make_pivot_clusters_batch)

                cost = 0.0
                inference_time = 0.0
                found_clusters = 0

                for data in test_loader:
                    data.to(device)
                    t0 = time.time()
                    out = model(data)
                    # normalisation is applied inside the helper
                    clustering = rounding_fn(out, data.edge_index, device)
                    t1 = time.time()
                    inference_time += t1 - t0
                    found_clusters += len(torch.unique(clustering))
                    cost += compute_cost_from_clustering_complete_graph(
                        clustering.to(device), data.edge_index
                    )

                curr_results = {
                    'ablation': ablation_name,
                    'iter': i,
                    'name': dataset_name,
                    'train_time': train_time,
                    'gpu_peak_memory_mb': gpu_peak_memory_mb,
                    'inference_time': inference_time,
                    'test_cost': cost,
                    'found_clusters': found_clusters,
                    'best_threshold': None,
                }

            else:
                # --- baseline / no_pivot_sampling / no_normalisation: percentile sweep ---
                best_threshold = None
                best_val_cost = float('inf')

                for threshold in np.arange(0.0, 1.01, 0.01):
                    val_cost = 0.0
                    for data in val_loader:
                        data.to(device)
                        out = model(data)
                        out = F.normalize(out, p=2, dim=1)
                        clustering = make_cc_clusters(
                            out, data.edge_index, threshold, device
                        )
                        val_cost += compute_cost_from_clustering_complete_graph(
                            clustering.to(device), data.edge_index
                        )
                    if val_cost < best_val_cost:
                        best_val_cost = val_cost
                        best_threshold = threshold

                cost = 0.0
                inference_time = 0.0
                found_clusters = 0
                for data in test_loader:
                    data.to(device)
                    t0 = time.time()
                    out = model(data)
                    out = F.normalize(out, p=2, dim=1)
                    clustering = make_cc_clusters(
                        out, data.edge_index, best_threshold, device
                    )
                    t1 = time.time()
                    inference_time += t1 - t0
                    found_clusters += len(torch.unique(clustering))
                    cost += compute_cost_from_clustering_complete_graph(
                        clustering.to(device), data.edge_index
                    )

                curr_results = {
                    'ablation': ablation_name,
                    'iter': i,
                    'name': dataset_name,
                    'train_time': train_time,
                    'gpu_peak_memory_mb': gpu_peak_memory_mb,
                    'inference_time': inference_time,
                    'test_cost': cost,
                    'found_clusters': found_clusters,
                    'best_threshold': best_threshold,
                }

            results.append(curr_results)
            mem_str = (f', GPU peak mem: {gpu_peak_memory_mb:.1f} MB'
                       if gpu_peak_memory_mb is not None else '')
            print(f'  [{ablation_name}] iter={i}  cost={cost:.2f}  '
                  f'train={train_time:.1f}s  infer={inference_time:.3f}s'
                  f'{mem_str}')

            # save after every iteration
            df = pd.DataFrame(results)
            df.to_csv(f'../results/ablation_linkgnn_{timestamp}.csv',
                      index=False)
            torch.cuda.empty_cache()

    # ---------- Print summary table ----------------------------------------
    df = pd.DataFrame(results)
    print('\n\n===== ABLATION RESULTS SUMMARY =====')
    summary = df.groupby('ablation')['test_cost'].agg(['mean', 'std', 'count'])
    print(summary.to_string())
    print(f'\nResults saved to ../results/ablation_linkgnn_{timestamp}.csv')


if __name__ == '__main__':
    main()
