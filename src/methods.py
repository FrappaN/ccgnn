from pyccalg import solve_instance
from mod_pivot import CorrelationClustering
from models import MLPModel, GNNModel, EMBMLPModel, EMBOnlyModel, LinearModel
import torch
import torch_geometric as tg
import numpy as np
import scipy.sparse as sp
import time
import scipy.optimize as opt

from torch_geometric.loader import NeighborLoader
from torch_geometric.data import Data
from torch_geometric.utils import dropout_edge, k_hop_subgraph
from train import train_nodemodel
from utils import compute_cost_from_clustering_complete_graph

from torch_geometric.utils import to_scipy_sparse_matrix
import random




def kwik_cluster(G):
    """
    Faster KwikCluster: uses set operations and precomputed neighbor sets
    to avoid repeated subgraph creation and expensive edge lookups.
    Returns a list of clusters (each cluster is a set of nodes).
    """
    nodes = set(G.nodes())
    if not nodes:
        return []

    # Precompute neighbor sets for O(1) membership tests and set intersections
    adj = {n: set(G.neighbors(n)) for n in nodes}

    clusters = []
    available = nodes  # nodes not yet assigned to a cluster

    while available:
        # pick a random pivot from the remaining nodes
        pivot = random.choice(tuple(available))

        # cluster = pivot plus all its neighbors that are still available
        cluster = {pivot}
        cluster |= (adj[pivot] & available)

        clusters.append(cluster)

        # remove clustered nodes from available set
        available -= cluster
    return clusters

def mod_pivot_method(G):
    seed = np.random.randint(0, 100000)
    cc = CorrelationClustering(G, seed)
    clusters = cc.get_clusters()
    # clusters is a defaultdict of sets, convert it to a list of sets
    clusters_list = list(clusters.values())
    return clusters_list


def r_pivot(G, r):
    # create a random permutation of the nodes
    nodes = list(G.nodes)
    np.random.shuffle(nodes)
    permutation = {node: i for i, node in enumerate(nodes)}

    unsettled = set(G.nodes)
    settled = set()
    clusters = []
    pivots = set()
    for _ in range(r):
        
        for v in unsettled:
            neighbors = set(G.neighbors(v))
            if not neighbors:
                continue
            if all(permutation[v] < permutation[u] for u in neighbors if u in unsettled):
                pivots.add(v)

        # mark pivots and their neighbors as settled
        for pivot in pivots:
            settled.add(pivot)
            unsettled.discard(pivot)
            for neighbor in G.neighbors(pivot):
                settled.add(neighbor)
                unsettled.discard(neighbor)

    non_pivots = set(G.nodes) - pivots
    # create clusters
    clusters = [{piv} for piv in pivots]  # start with the pivots cluster
    for u in non_pivots:
        neighbors = set(G.neighbors(u))
        pivots_in_neighbors = neighbors.intersection(pivots)
        if not pivots_in_neighbors:
            # no pivot in neighbors, u forms a singleton cluster
            clusters.append({u})
        elif any(all(permutation[neighbor] < permutation[v] for v in pivots_in_neighbors) for neighbor in neighbors if neighbor in unsettled):
            # unsettled neighbor with smaller rank than all pivots
            clusters.append({u})
        else:
            min_pivot = min(pivots_in_neighbors, key=lambda x: permutation[x])
            # join the cluster of the minimum rank pivot in neighbors
            for cluster in clusters:
                if min_pivot in cluster:
                    cluster.add(u)
                    break

    return clusters


def init_loss(objective='MinIntraDisagree'):
    if objective == 'MinIntraDisagree':
        def loss_fn(cluster_assignment_matrix, weight_matrix):
            # minimize intra-cluster agreement 
            trace = torch.einsum('ji,jk,ki->', cluster_assignment_matrix, weight_matrix, cluster_assignment_matrix)

            return -trace
    elif objective == 'MinInterDisagree':
        def loss_fn(cluster_assignment_matrix, weight_matrix):
            # maximize inter-cluster disagreement
            trace = torch.einsum('ji,jk,ki->', 1-cluster_assignment_matrix, weight_matrix, cluster_assignment_matrix)
            
            return trace
    elif objective == 'MinClustering':
        def loss_fn(cluster_assignment_matrix, weight_matrix):
            
            agreement_trace = torch.einsum('ji,jk,ki->', cluster_assignment_matrix, weight_matrix, cluster_assignment_matrix)
            disagreement_trace = torch.einsum('ji,jk,ki->', 1-cluster_assignment_matrix, weight_matrix, cluster_assignment_matrix)
            loss = disagreement_trace - agreement_trace

            return loss
    elif objective == 'MatrixFactorization':
        def loss_fn(cluster_assignment_matrix, input_tensor):
            co_cluster_matrix = torch.mm(cluster_assignment_matrix, cluster_assignment_matrix.t())
            dim1, dim2 = input_tensor.size()
            if dim1 == dim2:
                weight_matrix = input_tensor
            else:
                num_nodes = cluster_assignment_matrix.size(0)
                weight_matrix = torch.ones((num_nodes, num_nodes), device=input_tensor.device)*(-1.0)
                weight_matrix[input_tensor[0], input_tensor[1]] = 1.0
                weight_matrix[input_tensor[1], input_tensor[0]] = 1.0  # ensure symmetry
                weight_matrix[range(num_nodes), range(num_nodes)] = 0.0  # zero diagonal

            diff = torch.norm(weight_matrix - co_cluster_matrix, p='fro')**2
            norm = torch.norm(co_cluster_matrix, p='fro')**2
            loss = diff - norm
            return loss

    else:
        raise ValueError(f"Unknown objective: {objective}")
    
    return loss_fn


def method_switchcase(method, temp_data, objective='MinIntraDisagree', device='cpu', max_clusters=None, random_pivots=None, return_model=False, in_channels=2048, patience=100, lr=0.01, wd=5e-4, num_epochs=5000):

    if max_clusters is not None:
        out_clusters = min(max_clusters, temp_data.num_nodes)
    else:
        out_clusters = temp_data.num_nodes

    if method == 'MLP':

        model = MLPModel(
            in_channels=in_channels,
            hidden_channels=32, 
            out_channels=out_clusters,
            num_nodes=temp_data.num_nodes, 
            num_layers=2,
            no_gnn_train=False
            ).to(device)

    elif method == 'MLP-notrain':

        model = MLPModel(
            in_channels=in_channels,
            hidden_channels=32, 
            out_channels=out_clusters,
            num_nodes=temp_data.num_nodes, 
            num_layers=1,
            no_gnn_train=True
            ).to(device)

    elif method == 'GNN':

        model = GNNModel(
            in_channels=in_channels,
            hidden_channels=out_clusters,
            num_layers=2,
            ).to(device)

    elif method == 'EMBMLP':

        model = EMBMLPModel(
            hidden_channels=in_channels,
            out_channels=out_clusters,
            num_nodes=temp_data.num_nodes,
            num_layers=1
            ).to(device)

    elif method == 'EMBOnly':

        model = EMBOnlyModel(
            hidden_channels=out_clusters,
            num_nodes=temp_data.num_nodes,
            ).to(device)

    elif method == 'Linear':

        model = LinearModel(
            in_channels=in_channels,
            out_channels=out_clusters,
            ).to(device)
    
    elif method == 'kwikcluster':

        if getattr(temp_data, 'edge_weight', None) is not None:
            temp_data.edge_index = temp_data.edge_index[:, temp_data.edge_weight > 0]
            temp_data.edge_weight = temp_data.edge_weight[temp_data.edge_weight > 0]
        temp_data_nx = tg.utils.to_networkx(temp_data, to_undirected=True)

        runtime = time.time()
        kwik_partition = kwik_cluster(temp_data_nx)
        runtime = time.time() - runtime

        clustering = torch.ones(temp_data.num_nodes, dtype=torch.long)*(-1)
        for j, cluster in enumerate(kwik_partition):
            clustering[list(cluster)] = j

    elif method == 'r_pivot':

        if getattr(temp_data, 'edge_weight', None) is not None:
            temp_data.edge_index = temp_data.edge_index[:, temp_data.edge_weight > 0]
            temp_data.edge_weight = temp_data.edge_weight[temp_data.edge_weight > 0]
        temp_data_nx = tg.utils.to_networkx(temp_data, to_undirected=True)

        runtime = time.time()
        r_pivot_clusters = r_pivot(temp_data_nx, r=10000)
        runtime = time.time() - runtime

        clustering = torch.ones(temp_data.num_nodes, dtype=torch.long)*(-1)
        for j, cluster in enumerate(r_pivot_clusters):
            clustering[list(cluster)] = j

    elif method == 'random':
        # random clustering
        clustering = torch.randint(0, temp_data.num_nodes, (temp_data.num_nodes,))
        runtime = None

    elif method == 'original_partition':

        clustering = temp_data.partition
        runtime = None

    elif method == 'single_cluster':
        # single cluster cost
        clustering = torch.ones((temp_data.num_nodes))
        runtime = None

    elif method == 'modified_pivot':

        if getattr(temp_data, 'edge_weight', None) is not None:
            temp_data.edge_index = temp_data.edge_index[:, temp_data.edge_weight > 0]
            temp_data.edge_weight = temp_data.edge_weight[temp_data.edge_weight > 0]

        clustering = torch.ones(temp_data.num_nodes, dtype=torch.long)*-1
        adj = to_scipy_sparse_matrix(temp_data.edge_index, num_nodes=temp_data.num_nodes).tocoo()
        num_components, component = sp.csgraph.connected_components(adj)
        num_clusters = 0
        runtime = 0
        for i in range(num_components):
            subset_nodes = np.where(component == i)[0]
            subset_nodes = torch.tensor(subset_nodes, dtype=torch.long)
            _, component_edge_index, _, _ = k_hop_subgraph(subset_nodes, num_hops=1, edge_index=temp_data.edge_index, relabel_nodes=True, num_nodes=temp_data.num_nodes)
            if component_edge_index.size(1) == 0:
                # no edges in the component, each node is its own cluster
                clusters = [set([j]) for j in range(subset_nodes.size(0))]
            else:
                start_time = time.time()
                clusters = mod_pivot_method(component_edge_index)
                runtime += time.time() - start_time

            for j, cluster in enumerate(clusters):
                cluster_nodes = torch.tensor(list(cluster), dtype=torch.long).reshape(-1)
                cluster_nodes = subset_nodes[cluster_nodes].numpy()
                clustering[cluster_nodes] = j+num_clusters
            num_clusters += len(clusters)

    else:
        raise ValueError(f"Unknown method: {method}")

    if method in ['MLP', 'MLP-notrain', 'GNN', 'EMBMLP', 'EMBOnly', 'Linear']:

        loss_fn = init_loss(objective=objective)

        runtime = time.time()
        out = train_nodemodel(model, temp_data, loss_fn, lr, wd=wd, num_epochs=num_epochs, device=device, random_pivots=random_pivots, patience=patience)
        clustering = torch.argmax(out, dim=1)
        runtime = time.time() - runtime

    # first check all nodes are clustered
    assert not  (clustering == -1).any(), 'ERROR: NOT ALL NODES ARE CLUSTERED'

    cost = compute_cost_from_clustering_complete_graph(clustering, temp_data.edge_index)

    if return_model:
        return cost, runtime, clustering, model
    return cost, runtime, clustering