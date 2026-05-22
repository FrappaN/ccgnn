import os
os.environ['OMP_NUM_THREADS'] = '2'  # Limit the number of threads for parallel processing
os.environ['MKL_NUM_THREADS'] = '2'  # Limit the number of threads for MKL operations
os.environ['NUMEXPR_NUM_THREADS'] = '2'  # Limit the number of threads for NumExpr
os.environ['OPENBLAS_NUM_THREADS'] = '2'  # Limit the number of threads for OpenBLAS

import argparse
import torch
import pandas as pd
import numpy as np
import time
import torch.nn.functional as F
from tqdm import tqdm

from torch_geometric import seed_everything
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import ToUndirected, Compose, RemoveSelfLoops, RemoveIsolatedNodes
from torch_geometric.nn import Node2Vec
from ogb.graphproppred import PygGraphPropPredDataset

from methods import method_switchcase, init_loss

from load_utils import EXPWL1Dataset
from torch_geometric.datasets import TUDataset
from models import GNNModel, LinearModel
from utils import compute_cost_from_clustering_complete_graph
from train import train_linkmodel_batch, make_cc_clusters, train_nodemodel_batch

def add_ones(data):
    data.x = torch.ones((data.num_nodes,1), dtype=torch.float32)
    return data


def compute_node2vec_features_batch(dataset, n2v_dim=16, n2v_epochs=100,
                                     n2v_walk_length=10, n2v_context_size=10,
                                     n2v_lr=0.01, loader_batch_size=256):
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


# get day 
timestamp = time.strftime("%Y%m%d-%H%M%S")


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---- CLI arguments --------------------------------------------------------
parser = argparse.ArgumentParser(description='Inductive graph-level CC benchmark')
parser.add_argument('--n_iter', type=int, default=5, help='Number of iterations (default: 5)')
parser.add_argument('--patience', type=int, default=500, help='Patience for early stopping (default: 100)')
parser.add_argument('--max_clusters', type=int, default=None, help='Maximum number of clusters. If None, will be set to min(10000, num_nodes) (default: None)')
parser.add_argument('--num_epochs', type=int, default=5000, help='Number of training epochs (default: 5000)')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate (default: 0.01)')
parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay (default: 0.0)')
parser.add_argument('--datasets', type=str, nargs='+', default=None, 
                    help='List of datasets to run. Options: MUTAG, REDDIT-BINARY, NCI1, EXPWL1, github_stargazers, ogbg-ppa, ogbg-molhiv. If None, all datasets will be run (default: None)')
parser.add_argument('--methods', type=str, nargs='+', default=None,
                    help='List of methods to use. Options: GNN, LinkGNN, kwikcluster, modified_pivot, EMBOnly, LinkEMBOnly, Linear, LinearLink. If None, all methods will be used (default: None)')
parser.add_argument('--original', action='store_true', default=False,
                    help='Replace node features with original features')
parser.add_argument('--n2v_dim', type=int, default=128,
                    help='Node2Vec embedding dimension (default: 128)')
parser.add_argument('--n2v_epochs', type=int, default=100,
                    help='Node2Vec training epochs per graph (default: 100)')
parser.add_argument('--allones', action='store_true', default=False,
                    help='Replace node features with ones')

args = parser.parse_args()

if args.datasets is None:
    datasets = ['MUTAG', 'REDDIT-BINARY', 'NCI1', 'EXPWL1',  'github_stargazers', 'ogbg-ppa', 'ogbg-molhiv']
if args.methods is None:
    methods = ['GNN', 'LinkGNN', 'kwikcluster', 'modified_pivot', 'EMBOnly', 'LinkEMBOnly', 'Linear', 'LinearLink']
n_iters = args.n_iter
patience = args.patience
num_epochs = args.num_epochs
lr = args.lr
weight_decay = args.weight_decay
upper_max_clusters = args.max_clusters
costs_data = []


num_layers = 2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if args.allones and args.original:
    raise ValueError("Cannot use both --allones and --original at the same time.")



for dataset_name in datasets:
    transform = Compose([ToUndirected(), RemoveSelfLoops(), RemoveIsolatedNodes()])
    if dataset_name in ['MUTAG', 'NCI1', 'REDDIT-BINARY', 'COLLAB', 'PROTEINS', 'github_stargazers', 'REDDIT-MULTI-5K', 'DD', 'OHSU', 'FIRSTMM_DB']:
        if dataset_name in ['REDDIT-BINARY', 'COLLAB', 'github_stargazers', 'REDDIT-MULTI-5K', 'DD', 'OHSU']:
            transform = Compose([transform, add_ones])
            dataset = TUDataset(root='../data/TUDataset', name=dataset_name, use_node_attr=False, force_reload=False, transform=transform)
        else:
            dataset = TUDataset(root='../data/TUDataset', name=dataset_name, use_node_attr=True, force_reload=False)
    elif dataset_name in ['ogbg-molhiv', 'ogbg-ppa', 'ogbg-code2']:
        if dataset_name == 'ogbg-ppa':
            #if not args.node2vec:
            transform = Compose([transform, add_ones])
            dataset = PygGraphPropPredDataset(name=dataset_name, root='../data/OGB', transform=transform)
        elif dataset_name == 'ogbg-code2':
            dataset = PygGraphPropPredDataset(name=dataset_name, root='../data/OGB', transform=transform)
            dataset.data.x = dataset.data.x.to(torch.float32)
        else:
            dataset = PygGraphPropPredDataset(name=dataset_name, root='../data/OGB', transform=transform)
            dataset.data.x = dataset.data.x.to(torch.float32)

        # if dataset is ogbg-ppa, or ogbg-code2, restrict to 30.000 random graphs
        if dataset_name in ['ogbg-ppa', 'ogbg-code2']:
            perm = np.random.permutation(len(dataset))
            dataset = dataset[perm[:30000]]
    elif dataset_name == 'EXPWL1':
        if not args.node2vec:
            transform = Compose([transform, add_ones])
        dataset = EXPWL1Dataset(root=f'../data/{dataset_name}', transform=transform)
    else:
        raise ValueError(f'Unknown dataset: {dataset_name}')


    if dataset_name == 'REDDIT-MULTI-5K':
        batch_size = 32
    elif dataset_name in ['ogbg-ppa',  'ogbg-molhiv', 'ogbg-code2']:
        batch_size = 128
    else:
        batch_size = 64
    
    # ---- Optionally replace features with Node2Vec embeddings ----
    if args.original:
        feature_type = 'original'
    elif args.allones:
        feature_type = 'allones'
    else:
        feature_type = 'node2vec'
    
    methods = ['GNN', 'LinkGNN', 'modified_pivot', 'kwikcluster']  #  'GNN','scipy_opt',  'GNN', 'LinkGNN', 'kwikcluster', 'modified_pivot'  'kwikcluster', 'modified_pivot',
    objective = 'MatrixFactorization'

    max_clusters = max([data.num_nodes for data in dataset])
    if max_clusters*batch_size < upper_max_clusters:
        max_clusters = upper_max_clusters
    tot_nodes = sum([data.num_nodes for data in dataset])


    print('Loaded dataset:', dataset_name, 'max_clusters:', max_clusters,
          f'feature_type={feature_type}')
    for method in methods:
        if method == 'LinearLink' and dataset_name != 'REDDIT-BINARY':
            continue

        for i in range(n_iters):
            seed_everything(i*1000 + 6199)

            # random train/test split
            rnd_idx = np.random.permutation(len(dataset))

            split_idx = int(0.8 * len(dataset))
            train_dataset = [dataset[j] for j in rnd_idx[:split_idx]]
            val_dataset = [dataset[j] for j in rnd_idx[split_idx:int(0.9*len(dataset))]]
            test_dataset = [dataset[j] for j in rnd_idx[int(0.9*len(dataset)):]]

                
            # ---- Optionally replace features with Node2Vec embeddings ----
            if args.node2vec:
                print(f'[Node2Vec] Computing {args.n2v_dim}-dim embeddings for train_dataset with {len(train_dataset)} graphs, {args.n2v_epochs} epochs each ...')
                train_dataset = compute_node2vec_features_batch(
                    train_dataset,
                    n2v_dim=args.n2v_dim,
                    n2v_epochs=args.n2v_epochs,
                    loader_batch_size=batch_size,
                )
                print(f'[Node2Vec] Computing {args.n2v_dim}-dim embeddings for val_dataset with {len(val_dataset)} graphs, {args.n2v_epochs} epochs each ...')
                val_dataset = compute_node2vec_features_batch(
                    val_dataset,
                    n2v_dim=args.n2v_dim,
                    n2v_epochs=args.n2v_epochs,
                    loader_batch_size=batch_size,
                )
                print(f'[Node2Vec] Computing {args.n2v_dim}-dim embeddings for test_dataset with {len(test_dataset)} graphs, {args.n2v_epochs} epochs each ...')
                test_dataset = compute_node2vec_features_batch(
                    test_dataset,
                    n2v_dim=args.n2v_dim,
                    n2v_epochs=args.n2v_epochs,
                    loader_batch_size=batch_size,
                )
            elif args.allones:
                # perform a transform that replaces all node features with ones
                train_dataset = [add_ones(data) for data in train_dataset]
                val_dataset = [add_ones(data) for data in val_dataset]
                test_dataset = [add_ones(data) for data in test_dataset]

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            if method in ['GNN', 'LinkGNN', 'Linear', 'LinearLink']:
                in_channels = train_dataset[0].x.size(1)


                if method in ['GNN', 'Linear']:
                    out_channels = max_clusters
                else:
                    out_channels = 64

                if method in ['GNN', 'LinkGNN']:
                    model = GNNModel(in_channels=in_channels, hidden_channels=out_channels).to(device)
                else:
                    model = LinearModel(in_channels=in_channels, out_channels=out_channels).to(device)

            if method in ['GNN', 'Linear']:
                seed_everything(i*1000 + 6199)

                loss_fn = init_loss(objective)
                if device.type == 'cuda':
                    torch.cuda.reset_peak_memory_stats(device)
                train_time_start = time.time()
                model = train_nodemodel_batch(
                                            model,
                                            train_loader, 
                                            val_loader, 
                                            loss_fn=loss_fn, 
                                            device=device, 
                                            epochs=num_epochs, 
                                            lr=lr, 
                                            wd=weight_decay, 
                                            patience=patience, 
                                            batch_size=batch_size
                                        )
                train_time = time.time() - train_time_start
                gpu_peak_memory_mb = (
                    torch.cuda.max_memory_allocated(device) / 1024**2
                    if device.type == 'cuda' else None
                )

                # test on test set
                
                cost = 0.0
                inference_time = 0.0
                found_clusters = 0
                graphs_in_test = 0
                tot_nodes = 0
                tot_edges = 0
                for data in test_loader:
                    graphs_in_test += data.num_graphs
                    tot_nodes += data.num_nodes
                    tot_edges += data.num_edges
                    data.to(device)
                    time_start = time.time()
                    out = model(data)
                    cluster_assignment = torch.argmax(out, dim=1)
                    found_clusters += len(torch.unique(cluster_assignment))
                    time_end = time.time()
                    inference_time += time_end - time_start
                    cost += compute_cost_from_clustering_complete_graph(cluster_assignment, data.edge_index)
                    graphs_in_test+= data.num_graphs

                curr_results = {
                    'avg_nodes': tot_nodes/graphs_in_test, 
                    'avg_edges': tot_edges/graphs_in_test,
                    'name': dataset_name,
                    'method': method, 
                    'objective': objective,
                    'lr': lr,
                    'batch_size': batch_size,
                    'weight_decay': weight_decay,
                    'max_clusters': max_clusters,
                    'patience': patience,
                    'in_channels': in_channels,
                    'train_time': train_time,
                    'gpu_peak_memory_mb': gpu_peak_memory_mb,
                    'inference_time': inference_time,
                    'test_cost': cost,
                    'found_clusters': found_clusters,
                    'feature_type': feature_type,
                    'n2v_dim': args.n2v_dim,
                    'n2v_epochs': args.n2v_epochs,
                    'graphs_in_test': graphs_in_test,
                }   

                print(f"Method: {method}, {objective}, Cost: {cost}, Runtime: {inference_time:.2f} seconds, Train time: {train_time:.2f}s, GPU peak mem: {gpu_peak_memory_mb:.1f} MB" if gpu_peak_memory_mb is not None else f"Method: {method}, {objective}, Cost: {cost}, Runtime: {inference_time:.2f} seconds, Train time: {train_time:.2f}s")

            elif method in ['LinkGNN', 'LinearLink']:
                seed_everything(6199+i*1000)

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
                                            batch_size=batch_size
                                        )
                train_time = time.time() - train_time_start
                gpu_peak_memory_mb = (
                    torch.cuda.max_memory_allocated(device) / 1024**2
                    if device.type == 'cuda' else None
                )

                # first select best threshold on train graph
                best_threshold = None
                best_train_cost = float('inf')
                for threshold in np.arange(0.0, 1.01, 0.01):
                    train_cost = 0.0
                    for data in val_loader:
                        data.to(device)
                        out = model(data)
                        out = F.normalize(out, p=2, dim=1)

                        clustering = make_cc_clusters(out, data.edge_index, threshold, device)
                        
                        train_cost += compute_cost_from_clustering_complete_graph(clustering.to(device), data.edge_index)

                    if train_cost < best_train_cost:
                        best_threshold = threshold
                        best_train_cost = train_cost

                
                cost = 0.0
                inference_time = 0.0
                found_clusters = 0
                graphs_in_test = 0
                tot_nodes = 0
                tot_edges = 0
                for data in test_loader:
                    data.to(device)
                    time_start = time.time()
                    out = model(data)
                    out = F.normalize(out, p=2, dim=1)
                    clustering = make_cc_clusters(out, data.edge_index, best_threshold, device)
                    time_end = time.time()
                    found_clusters += len(torch.unique(clustering))
                    cost += compute_cost_from_clustering_complete_graph(clustering.to(device), data.edge_index)
                    graphs_in_test += data.num_graphs
                    inference_time += time_end - time_start
                    tot_nodes += data.num_nodes
                    tot_edges += data.num_edges

                curr_results = {
                    'avg_nodes': tot_nodes/graphs_in_test, 
                    'avg_edges': tot_edges/graphs_in_test,
                    'name': dataset_name,
                    'method': method,
                    'max_clusters': None,
                    'objective': 'LinkObjective',
                    'lr': lr,
                    'batch_size': batch_size,
                    'weight_decay': weight_decay,
                    'in_channels': in_channels,
                    'patience': patience,
                    'train_time': train_time,
                    'gpu_peak_memory_mb': gpu_peak_memory_mb,
                    'found_clusters': found_clusters,
                    'test_cost': cost, 
                    'best_threshold': best_threshold,
                    'inference_time': inference_time,
                    'graphs_in_test': graphs_in_test,
                    'feature_type': feature_type,
                    'n2v_dim': args.n2v_dim,
                    'n2v_epochs': args.n2v_epochs,
                    }

                print(f"Method: {method}, Cost: {cost}, Runtime: {inference_time:.2f} seconds, Train time: {train_time:.2f}s, GPU peak mem: {gpu_peak_memory_mb:.1f} MB" if gpu_peak_memory_mb is not None else f"Method: {method}, Cost: {cost}, Runtime: {inference_time:.2f} seconds, Train time: {train_time:.2f}s")

            else:
                seed_everything(i*1000 + 6199)
                cost = 0.0
                inference_time = 0.0
                found_clusters = 0
                tot_nodes = 0
                tot_edges = 0
                graphs_in_test = 0
                for data in test_loader:
                    graphs_in_test += data.num_graphs
                    tot_nodes += data.num_nodes
                    tot_edges += data.num_edges
                    curr_cost, curr_runtime, clustering = method_switchcase(method, data, device=device, max_clusters=max_clusters)
                    cost += curr_cost
                    inference_time += curr_runtime
                    found_clusters += len(torch.unique(clustering))

                curr_results = {
                    'avg_nodes': tot_nodes/graphs_in_test, 
                    'avg_edges': tot_edges/graphs_in_test,
                    'name': dataset_name,
                    'method': method,
                    'max_clusters': data.num_nodes,
                    'objective': 'MinDisagree',
                    'in_channels': None,
                    'patience': None,
                    'found_clusters': found_clusters,
                    'test_cost': cost, 
                    'inference_time': inference_time,
                    }
            
                print(f"Method: {method}, Cost: {cost}, Runtime: {inference_time:.2f} seconds")
            print(f"Average number of clusters: {found_clusters/graphs_in_test}, Average number of nodes: {tot_nodes/graphs_in_test}, Average number of edges: {tot_edges/graphs_in_test}")
            costs_data.append(curr_results)
            # save temporary results
            costs_data_df = pd.DataFrame(costs_data)
            costs_data_df.to_csv(f'../results/costs_inductive_{timestamp}.csv', index=False)
            torch.cuda.empty_cache()


