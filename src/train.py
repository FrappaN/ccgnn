import torch
from torch_geometric.data import Data
import numpy as np
from torch_geometric.utils import k_hop_subgraph, to_networkx
from torch_geometric.utils import to_scipy_sparse_matrix
from tqdm import tqdm

import scipy.sparse as sp



def _signed_weight_matrix(num_nodes, edge_index, device):
    """W with +1 on edges, -1 off-edges, 0 on the diagonal."""
    W = torch.ones((num_nodes, num_nodes), device=device) * (-1.0)
    W[edge_index[0], edge_index[1]] = 1.0
    W[edge_index[1], edge_index[0]] = 1.0  # ensure symmetry
    W.fill_diagonal_(0.0)
    return W


def train_linkmodel(model, data, device, epochs=100, lr=0.01, weight_decay=0.0,
                    patience=10, random_pivots=1000):
    model = model.to(device)

    pos_edge_mask = getattr(data, 'edge_weight', None)
    if pos_edge_mask is not None:
        pos_edge_mask = pos_edge_mask > 0
        edge_index = data.edge_index[:, pos_edge_mask]
    else:
        edge_index = data.edge_index

    weight_matrix = getattr(data, 'W', None)

    if random_pivots is None:
        data.edge_index = edge_index
        data.to(device)
        if weight_matrix is not None:
            curr_weight_matrix = weight_matrix.to(device)
        else:
            curr_weight_matrix = _signed_weight_matrix(data.num_nodes, edge_index, device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_loss = float('inf')
    # FIX: previously only assigned inside `if epoch >= patience`, so a run with
    # num_epochs <= patience ended in UnboundLocalError at load_state_dict.
    best_model = {k: v.detach().clone() for k, v in model.state_dict().items()}
    count = 0

    loss_list = []

    model.train()
    for epoch in (pbar := tqdm(range(epochs), desc="Training")):

        optimizer.zero_grad()
        if random_pivots is not None:
            temp_data = Data()

            # use as edge_index only the edges between a random subset of pivots and all other nodes
            pivots = np.random.choice(data.num_nodes, size=min(random_pivots, data.num_nodes), replace=False).tolist()

            nodes, pivot_edge_index, _, pivot_edge_mask = k_hop_subgraph(pivots, 1, edge_index, relabel_nodes=True, num_nodes=data.num_nodes)

            temp_data.x = data.x[nodes].to(device)

            # check if model is Linear or GNN
            if isinstance(model, torch.nn.Linear):
                out = model(temp_data.x)
            else:
                temp_data.edge_index = pivot_edge_index.to(device)
                out = model(temp_data)

            if weight_matrix is not None:
                curr_weight_matrix = weight_matrix[nodes, :][:, nodes].to(device)
            else:
                curr_weight_matrix = _signed_weight_matrix(len(nodes), pivot_edge_index, out.device)
        else:
            out = model(data)

        out = torch.nn.functional.normalize(out, p=2, dim=1)

        dists = torch.cdist(out, out) / 2
        co_cluster_matrix = 1 - dists

        diff = torch.norm(curr_weight_matrix - co_cluster_matrix, p='fro')**2
        norm = torch.norm(co_cluster_matrix, p='fro')**2
        loss = diff - norm

        loss.backward()
        optimizer.step()

        loss_list.append(loss.item())

        if random_pivots is not None:
            avg_loss = np.mean(loss_list[-30:])
            del temp_data, nodes, pivot_edge_index
        else:
            avg_loss = loss.item()

        del out, loss
        # clear cache every ten epochs
        if epoch % 10 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

        if epoch >= patience:
            if avg_loss <= best_loss - 1:
                best_loss = avg_loss
                best_model = {k: v.detach().clone() for k, v in model.state_dict().items()}
                count = 0
            else:
                count += 1
                if count == patience:
                    break
    model.load_state_dict(best_model)
    model.eval()
    return model


def train_linkmodel_batch(model, train_loader, val_loader, device, epochs=100,
                          lr=0.01, wd=0.0, patience=10, batch_size=64):
    """
    Mini-batch (inductive) counterpart of `train_linkmodel`.

    NOTE: this function was referenced by main_inductive.py and threshold_sweep.py
    but was missing from the cleaned repo, so both scripts failed at import.
    """
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    best_loss = float('inf')
    best_model = {k: v.detach().clone() for k, v in model.state_dict().items()}
    count = 0

    for epoch in (pbar := tqdm(range(epochs), desc="Training")):
        model.train()
        for data in train_loader:
            m = data.num_nodes
            edge_index = data.edge_index
            weight_matrix = getattr(data, 'W', None)

            # filter to positive edges only if edge_weight is present
            pos_edge_mask = getattr(data, 'edge_weight', None)
            if pos_edge_mask is not None:
                pos_edge_mask = pos_edge_mask > 0
                edge_index = edge_index[:, pos_edge_mask]

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
            out = torch.nn.functional.normalize(out, p=2, dim=1)

            if weight_matrix is not None:
                curr_W = weight_matrix[subgraph_nodes, :][:, subgraph_nodes].to(device)
            else:
                curr_W = _signed_weight_matrix(num_subgraph_nodes,
                                               subgraph_edge_index.to(out.device),
                                               out.device)

            dists = torch.cdist(out, out) / 2
            co_cluster_matrix = 1 - dists
            diff = torch.norm(curr_W - co_cluster_matrix, p='fro')**2
            norm = torch.norm(co_cluster_matrix, p='fro')**2
            loss = diff - norm

            loss.backward()
            optimizer.step()
            del subgraph_data, subgraph_nodes, subgraph_edge_index, out, curr_W, loss

        if epoch % 10 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

        tot_loss = 0.0
        model.eval()
        with torch.no_grad():
            for val_data in val_loader:
                val_data = val_data.to(device)
                m = val_data.num_nodes
                val_edge_index = val_data.edge_index
                val_weight_matrix = getattr(val_data, 'W', None)

                val_out = torch.nn.functional.normalize(model(val_data), p=2, dim=1)

                if val_weight_matrix is not None:
                    curr_W = val_weight_matrix.to(device)
                else:
                    curr_W = _signed_weight_matrix(m, val_edge_index, val_out.device)

                val_dists = torch.cdist(val_out, val_out) / 2
                val_co = 1 - val_dists
                tot_loss += (torch.norm(curr_W - val_co, p='fro')**2
                             - torch.norm(val_co, p='fro')**2).item()

        if tot_loss <= best_loss - 1:
            best_loss = tot_loss
            best_model = {k: v.detach().clone() for k, v in model.state_dict().items()}
            count = 0
        elif epoch >= patience:
            count += 1
            if count == patience:
                break

    model.load_state_dict(best_model)
    model.eval()
    return model


def train_nodemodel(model, data, loss_fn, lr, wd=5e-4, num_epochs=100, patience=100, random_pivots=None, device='cpu'):
    model.train()

    loss_list = []

    best_loss = float('inf')
    best_model = {k: v.detach().clone() for k, v in model.state_dict().items()}

    count = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    pos_edge_mask = getattr(data, 'edge_weight', None)
    if pos_edge_mask is not None:
        pos_edge_mask = pos_edge_mask > 0
        edge_index = data.edge_index[:, pos_edge_mask]
    else:
        edge_index = data.edge_index

    if random_pivots is None:
        data.edge_index = edge_index.to(device)
        data.to(device)
        if getattr(data, 'W', None) is not None:
            data.W = data.W.to(device)

    num_pivots = random_pivots if random_pivots is not None else None

    for epoch in (pbar := tqdm(range(num_epochs), desc="Training", unit="epoch")):

        optimizer.zero_grad()
        if random_pivots is not None:
            temp_data = Data()

            # use as edge_index only the edges between a random subset of pivots and all other nodes
            pivots = np.random.choice(data.num_nodes, size=min(num_pivots, data.num_nodes), replace=False).tolist()

            nodes, pivot_edge_index, _, pivot_edge_mask = k_hop_subgraph(pivots, 1, edge_index, relabel_nodes=True, num_nodes=data.num_nodes)

            temp_data.x = data.x[nodes].to(device)
            temp_data.edge_index = pivot_edge_index.to(device)
            temp_data.edge_weight = data.edge_weight[pivot_edge_mask].to(device) if getattr(data, 'edge_weight', None) is not None else None

            out = model(temp_data)

            cluster_assignment_matrix = torch.softmax(out, dim=1)

            # only preserve non-zero columns
            non_zero_cols = torch.where(cluster_assignment_matrix.sum(dim=0) != 0)[0]

            # only calculate the loss on the subgraph induced by the pivots
            if getattr(data, 'W', None) is not None:
                weight_matrix = data.W[nodes.cpu(), :][:, nodes.cpu()].to(device)
            cluster_assignment_matrix = cluster_assignment_matrix[:, non_zero_cols]

            if getattr(data, 'W', None) is not None:
                loss = loss_fn(cluster_assignment_matrix, weight_matrix)
            else:
                loss = loss_fn(cluster_assignment_matrix, temp_data.edge_index)

            loss.backward()
            optimizer.step()
            del temp_data, nodes, pivot_edge_index

        else:
            out = model(data)
            cluster_assignment_matrix = torch.softmax(out, dim=1)
            # only preserve non-zero columns
            non_zero_cols = torch.where(cluster_assignment_matrix.sum(dim=0) != 0)[0]
            cluster_assignment_matrix = cluster_assignment_matrix[:, non_zero_cols]
            if getattr(data, 'W', None) is not None:
                loss = loss_fn(cluster_assignment_matrix, data.W)
            else:
                loss = loss_fn(cluster_assignment_matrix, data.edge_index)

            loss.backward()
            optimizer.step()

        loss_list.append(loss.item())

        if random_pivots is not None:
            avg_loss = np.mean(loss_list[-30:])
        else:
            avg_loss = loss.item()

        if epoch >= patience:
            if avg_loss <= best_loss - 1:
                best_loss = avg_loss
                best_model = {k: v.detach().clone() for k, v in model.state_dict().items()}
                count = 0
            else:
                count += 1
                if count == patience:
                    break

        del out, cluster_assignment_matrix, loss
        if epoch % 100 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    model.load_state_dict(best_model)

    with torch.no_grad():
        model.eval()

        if random_pivots is None:
            best_out = model(data).to('cpu')
        else:
            batches_indices = torch.arange(0, data.num_nodes, step=random_pivots)
            all_out = []
            for start in batches_indices:
                end = min(start + random_pivots, data.num_nodes)
                batch_nodes = torch.arange(start, end)
                batch_data = Data()
                full_node_batch, batch_edge_index, inv_node_map, _ = k_hop_subgraph(batch_nodes, 1, edge_index, relabel_nodes=True, num_nodes=data.num_nodes)
                batch_data.x = data.x[full_node_batch.cpu()].to(device)
                batch_data.edge_index = batch_edge_index.to(device)
                batch_out = model(batch_data)[inv_node_map]
                all_out.append(batch_out.cpu())
                del batch_data, batch_edge_index, batch_out, batch_nodes, full_node_batch, inv_node_map
            best_out = torch.cat(all_out, dim=0).to('cpu')
            del all_out, batches_indices
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    del model, data
    return best_out


def train_nodemodel_batch(model, train_loader, val_loader, loss_fn, lr=0.01,
                          wd=5e-4, epochs=100, patience=100, device='cpu',
                          batch_size=64):
    """
    Mini-batch (inductive) counterpart of `train_nodemodel`.

    NOTE: referenced by main_inductive.py but missing from the cleaned repo.
    """
    model = model.to(device)

    best_loss = float('inf')
    best_model = {k: v.detach().clone() for k, v in model.state_dict().items()}
    count = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    for epoch in (pbar := tqdm(range(epochs), desc="Training", unit="epoch")):
        model.train()
        for data in train_loader:
            m = data.num_nodes

            # filter to positive edges only if edge_weight is present
            pos_edge_mask = getattr(data, 'edge_weight', None)
            if pos_edge_mask is not None:
                pos_edge_mask = pos_edge_mask > 0
                edge_index = data.edge_index[:, pos_edge_mask]
            else:
                edge_index = data.edge_index

            weight_matrix = getattr(data, 'W', None)

            optimizer.zero_grad()

            pivots = torch.randperm(m)[:batch_size].tolist()

            subgraph_nodes, subgraph_edge_index, _, subgraph_edge_mask = k_hop_subgraph(
                pivots, 1, edge_index, relabel_nodes=True, num_nodes=m)

            subgraph_data = Data()
            subgraph_data.x = data.x[subgraph_nodes.cpu()].to(device)
            subgraph_data.edge_index = subgraph_edge_index.to(device)
            subgraph_data.edge_weight = (
                data.edge_weight[subgraph_edge_mask].to(device)
                if getattr(data, 'edge_weight', None) is not None else None
            )

            out = model(subgraph_data)
            cluster_assignment_matrix = torch.softmax(out, dim=1)

            # only preserve non-zero columns
            non_zero_cols = torch.where(cluster_assignment_matrix.sum(dim=0) != 0)[0]
            cluster_assignment_matrix = cluster_assignment_matrix[:, non_zero_cols]

            if weight_matrix is not None:
                sub_W = weight_matrix[subgraph_nodes.cpu(), :][:, subgraph_nodes.cpu()].to(device)
                loss = loss_fn(cluster_assignment_matrix, sub_W)
            else:
                loss = loss_fn(cluster_assignment_matrix, subgraph_data.edge_index)

            loss.backward()
            optimizer.step()

            del subgraph_data, subgraph_nodes, subgraph_edge_index, out, cluster_assignment_matrix, loss

        if epoch % 10 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

        tot_loss = 0.0
        model.eval()
        with torch.no_grad():
            for val_data in val_loader:
                val_data = val_data.to(device)
                val_out = model(val_data)
                val_cluster_assignment_matrix = torch.softmax(val_out, dim=1)

                non_zero_cols = torch.where(val_cluster_assignment_matrix.sum(dim=0) != 0)[0]
                val_cluster_assignment_matrix = val_cluster_assignment_matrix[:, non_zero_cols]

                if getattr(val_data, 'W', None) is not None:
                    val_loss = loss_fn(val_cluster_assignment_matrix, val_data.W)
                else:
                    val_loss = loss_fn(val_cluster_assignment_matrix, val_data.edge_index)
                tot_loss += val_loss.item()

        if tot_loss <= best_loss - 1:
            best_loss = tot_loss
            best_model = {k: v.detach().clone() for k, v in model.state_dict().items()}
            count = 0
        elif epoch >= patience:
            count += 1
            if count == patience:
                break

    model.load_state_dict(best_model)
    model.eval()
    return model


def compute_out(model, data, device, batch_size=1000):
    clear_counter = 0
    with torch.no_grad():
        if batch_size is None:
            model.eval()
            if isinstance(model, torch.nn.Linear):
                out = model(data.x.to(device))
            else:
                out = model(data.to(device))
            out = torch.nn.functional.normalize(out, p=2, dim=1)

        else:
            batches_indices = torch.arange(0, data.num_nodes, step=batch_size)
            all_outs = []
            for start in batches_indices:
                end = min(start + batch_size, data.num_nodes)
                batch_nodes = torch.arange(start, end)
                batch_data = Data()

                if isinstance(model, torch.nn.Linear):
                    batch_out = model(data.x[batch_nodes].to(device))
                else:
                    full_node_batch, batch_edge_index, inv_node_map, _ = k_hop_subgraph(batch_nodes, 1, data.edge_index, relabel_nodes=True, num_nodes=data.num_nodes)
                    batch_data.x = data.x[full_node_batch.cpu()].to(device)
                    batch_data.edge_index = batch_edge_index.to(device)
                    batch_out = model(batch_data)[inv_node_map]
                    del batch_data, batch_edge_index, full_node_batch, inv_node_map

                batch_out = torch.nn.functional.normalize(batch_out, p=2, dim=1)
                all_outs.append(batch_out.cpu())
                del batch_out, batch_nodes
                clear_counter += 1
                if clear_counter % 1000 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            out = torch.cat(all_outs, dim=0)
            del all_outs, batches_indices
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return out


def make_cc_clusters(out, edge_index, threshold, device):
    """
    Connected-component clustering: keep the (1 - threshold) fraction of edges
    with the smallest embedding distance, then take connected components.
    """
    num_nodes = out.size(0)
    undirected_edge_index = edge_index[:, edge_index[0] < edge_index[1]].to(device)

    out_half = out.to(device)
    row_out = out_half[undirected_edge_index[0]]
    col_out = out_half[undirected_edge_index[1]]
    dists_sq = torch.sum((row_out - col_out) ** 2, dim=1)

    k = max(int((1 - threshold) * dists_sq.size(0)), 1)
    topk_vals, topk_indices = torch.topk(dists_sq, k, largest=False)
    filtered_edges = undirected_edge_index[:, topk_indices]

    filtered_edges = torch.cat([filtered_edges, filtered_edges[[1, 0], :]], dim=1)

    sp_graph = to_scipy_sparse_matrix(filtered_edges.cpu(), num_nodes=num_nodes)
    _, labels = sp.csgraph.connected_components(sp_graph, directed=False, return_labels=True)
    clustering = torch.from_numpy(labels).long()

    del row_out, col_out, dists_sq, undirected_edge_index, filtered_edges, sp_graph, labels
    return clustering


def make_pivot_clusters(out, device, threshold=None, seed=6199):
    """
    Pivot rounding on the embedding metric.  With `threshold=None` the radius is
    drawn uniformly at random per node, which is the randomised LP rounding;
    with a fixed threshold it is the deterministic ball variant.

    NOTE: referenced by ablation_ind.py but missing from the released code.
    Distances are NOT rescaled here, so a fixed absolute threshold will
    over-merge; ablation_ind.py rescales before calling this.
    """
    num_nodes = out.size(0)

    pivot_clustering = torch.ones(num_nodes, dtype=torch.long) * -1
    rng = np.random.default_rng(seed=seed)
    unclustered_nodes = list(range(num_nodes))
    k = 0
    while len(unclustered_nodes) > 0:
        pivot = int(rng.choice(unclustered_nodes))
        unclustered_nodes.remove(pivot)
        if not unclustered_nodes:
            pivot_clustering[pivot] = k
            k += 1
            break
        array_unclustered = torch.tensor(unclustered_nodes, dtype=torch.long)
        dists = torch.cdist(out[pivot:pivot + 1].to(device),
                            out[array_unclustered].to(device)).cpu()
        dists = dists / 2
        if threshold is None:
            random_rngs = torch.tensor(rng.random(len(unclustered_nodes)),
                                       dtype=dists.dtype)
            cluster = array_unclustered[dists[0, :] <= random_rngs].tolist()
        else:
            cluster = array_unclustered[dists[0, :] <= threshold].tolist()
        for node in cluster:
            unclustered_nodes.remove(node)
        cluster = set(cluster + [pivot])

        pivot_clustering[list(cluster)] = k
        k += 1
        del dists, array_unclustered

    return pivot_clustering
