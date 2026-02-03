import torch
from torch_geometric.data import Data
import numpy as np
from torch_geometric.utils import k_hop_subgraph, to_networkx
from torch_geometric.utils import dropout_edge, to_scipy_sparse_matrix
from tqdm import tqdm
from utils import round_charikar
import networkx as nx
import scipy.sparse as sp
import itertools
from utils import compute_cost_from_clustering_complete_graph, compute_cost_from_clustering


def train_linkmodel(model, data, device, epochs=100, lr=0.01, weight_decay=0.0, patience=10, random_pivots=1000):
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
            num_nodes = data.num_nodes
            curr_weight_matrix = torch.ones((num_nodes, num_nodes), device=device)*(-1.0)
            curr_weight_matrix[edge_index[0], edge_index[1]] = 1.0
            curr_weight_matrix[edge_index[1], edge_index[0]] = 1.0  # ensure symmetry
            curr_weight_matrix[range(num_nodes), range(num_nodes)] = 0.0  # zero diagonal

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_loss = float('inf')
    count = 0

    loss_list = []

    model.train()
    for epoch in (pbar:=tqdm(range(epochs), desc="Training")):

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
                num_nodes = len(nodes)
                curr_weight_matrix = torch.ones((num_nodes, num_nodes), device=out.device)*(-1.0)
                curr_weight_matrix[pivot_edge_index[0], pivot_edge_index[1]] = 1.0
                curr_weight_matrix[pivot_edge_index[1], pivot_edge_index[0]] = 1.0  # ensure symmetry
                curr_weight_matrix[range(num_nodes), range(num_nodes)] = 0.0  # zero diagonal
        else:
            out = model(data)



        out = torch.nn.functional.normalize(out, p=2, dim=1)
        # alterative: torch.nn.functional.tanh(out)

        dists = torch.cdist(out, out)
        dists = dists / 2

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

        del out, loss, diff, norm, dists #  clustering, sims
        # clear cache every ten epochs
        if epoch % 10 == 0:
            torch.cuda.empty_cache()


        if epoch >= patience:
            if avg_loss <= best_loss-1:
                best_loss = avg_loss
                best_model = model.state_dict()
                count = 0
            else:
                count += 1
                if count == patience:
                    break
    model.load_state_dict(best_model)
    model.eval()
    return model


def train_nodemodel(model, data, loss_fn, lr, wd=5e-4, num_epochs=100, patience=100, random_pivots=None, device='cpu', dropout=False):
    model.train()

    loss_list = []

    best_loss = float('inf')

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
            temp_data.edge_weight = data.edge_weight[pivot_edge_mask].to(device) if data.edge_weight is not None else None
            


            out = model(temp_data)

            cluster_assignment_matrix = torch.softmax(out, dim=1)
            
            # only preserve non-zero columns
            non_zero_cols = torch.where(cluster_assignment_matrix.sum(dim=0) != 0)[0]

            # only calculate the loss on the subgraph induced by the pivots
            if getattr(data, 'W', None) is not None:
                weight_matrix = data.W[nodes.cpu(), :][:, nodes.cpu()].to(device)
            cluster_assignment_matrix = cluster_assignment_matrix[:, non_zero_cols]#/len(nodes) # normalize by number of nodes in subgraph

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
            avg_loss = loss

        if epoch >= patience:
            if avg_loss <= best_loss-1:
                best_loss = avg_loss
                best_model = model.state_dict()
                count = 0
            else:
                count += 1
                if count == patience:
                    break

        del out, cluster_assignment_matrix, loss
        if epoch % 100 == 0:
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
            torch.cuda.empty_cache()

        # cluster_assignment = torch.argmax(best_out, dim=1)
        # best_clustering = torch.nn.functional.one_hot(cluster_assignment).float()
    del model, data
    return best_out





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
            #sims = out @ cluster_embs.T

        else:
            batches_indices = torch.arange(0, data.num_nodes, step=batch_size)
            all_sims = []
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
                    del batch_data, batch_edge_index,full_node_batch, inv_node_map

                batch_out = torch.nn.functional.normalize(batch_out, p=2, dim=1)
                all_outs.append(batch_out.cpu())
                #batch_sims = batch_out @ cluster_embs.T
                #all_sims.append(batch_sims.cpu())
                del  batch_out, batch_nodes
                clear_counter += 1
                if clear_counter % 1000 == 0:
                    torch.cuda.empty_cache()
            #sims = torch.cat(all_sims, dim=0)
            out = torch.cat(all_outs, dim=0)
            del all_sims, all_outs, batches_indices
            torch.cuda.empty_cache()
    return out


def make_cc_clusters(out, edge_index, threshold, device):
    """
    Optimized connected-component style clustering using precomputed distances and GPU acceleration.
    - Precomputes all pairwise distances once on the GPU.
    - Assumes a complete graph for traversal.
    """
    num_nodes = out.size(0)
    undirected_edge_index = edge_index[:, edge_index[0] < edge_index[1]].to(device)

    # Precompute pairwise distances on GPU
    out_half = out.to(device)
    row_out = out_half[undirected_edge_index[0]]
    col_out = out_half[undirected_edge_index[1]]
    dists_sq = torch.sum((row_out - col_out) ** 2, dim=1)


    # thr_sq = torch.quantile(dists_sq, 1-pool_ratio)
    # filtered_edges = undirected_edge_index[:, dists_sq <= thr_sq]
    # alternative: take only the pool_ratio closest edges

    k = max(int((1-threshold) * dists_sq.size(0)), 1)
    topk_vals, topk_indices = torch.topk(dists_sq, k, largest=False)
    filtered_edges = undirected_edge_index[:, topk_indices]

    filtered_edges = torch.cat([filtered_edges, filtered_edges[[1,0],:]], dim=1)  # make directed
    
    
    sp_graph = to_scipy_sparse_matrix(filtered_edges, num_nodes=num_nodes)
    _, labels = sp.csgraph.connected_components(sp_graph, directed=False, return_labels=True)
    clustering = torch.from_numpy(labels).long()

    del row_out, col_out, dists_sq, undirected_edge_index, filtered_edges, sp_graph, labels
    return clustering


