import os
os.environ['OMP_NUM_THREADS'] = '2'  # Limit the number of threads for parallel processing
os.environ['MKL_NUM_THREADS'] = '2'  # Limit the number of threads for MKL operations
os.environ['NUMEXPR_NUM_THREADS'] = '2'  # Limit the number of threads for NumExpr
os.environ['OPENBLAS_NUM_THREADS'] = '2'  # Limit the number of threads for OpenBLAS

import torch
import pandas as pd
import numpy as np
import time
import argparse
from sklearn.metrics import normalized_mutual_info_score
from utils import compute_cost_from_clustering_complete_graph, eval_pairwise_accuracy

from torch_geometric import seed_everything



from methods import method_switchcase

from load_utils import load_suitesparse_dataset, load_Planetoid_dataset, \
                       load_Amazon_dataset, load_OGBN_dataset, load_wikics_dataset, \
                       load_github_dataset, load_facebook_dataset
from models import GNNModel, LinearModel, EMBOnlyModel
from train import train_linkmodel, make_cc_clusters, compute_out


# get day 
timestamp = time.strftime("%Y%m%d-%H%M%S")


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Parse command-line arguments
parser = argparse.ArgumentParser(description='Run correlation clustering experiments on graph datasets.')
parser.add_argument('--n_iter', type=int, default=5, help='Number of iterations (default: 5)')
parser.add_argument('--patience', type=int, default=100, help='Patience for early stopping (default: 100)')
parser.add_argument('--random_pivots', type=int, default=1000, help='Number of random pivots (default: 1000)')
parser.add_argument('--max_clusters', type=int, default=None, help='Maximum number of clusters. If None, will be set to min(10000, num_nodes) (default: None)')
parser.add_argument('--in_channels', type=int, default=512, help='Number of input channels for embeddings (default: 512)')
parser.add_argument('--num_epochs', type=int, default=5000, help='Number of training epochs (default: 5000)')
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate (default: 0.01)')
parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay (default: 0.0)')
parser.add_argument('--batch_size', type=int, default=5000, help='Batch size for LinkGNN model inference (default: 5000)')
parser.add_argument('--datasets', type=str, nargs='+', default=None, 
                    help='List of datasets to run. Options: polblogs, ca-GrQc, ca-HepTh, ca-AstroPh, email-Enron, cond-mat-2005, Cora, PubMed, CiteSeer, AmazonPhoto, AmazonComputers, WikiCS, FacebookPagePage, GitHub, OGBN-Arxiv. If None, all datasets will be run (default: None)')
parser.add_argument('--methods', type=str, nargs='+', default=None,
                    help='List of methods to use. Options: GNN, LinkGNN, kwikcluster, modified_pivot, EMBOnly, LinkEMBOnly, Linear, LinearLink. If None, all methods will be used (default: None)')

args = parser.parse_args()

# Define available datasets
cc_bench_datasets = [('Newman', 'polblogs'), ('SNAP', 'ca-GrQc'), ('SNAP', 'ca-HepTh'), ('SNAP', 'ca-AstroPh'), ('SNAP', 'email-Enron'), ('Newman', 'cond-mat-2005')]
feature_datasets = ['Cora', 'PubMed', 'CiteSeer', 'AmazonPhoto', 'AmazonComputers', 'WikiCS', 'FacebookPagePage', 'GitHub',  'OGBN-Arxiv'] 
all_datasets = cc_bench_datasets + feature_datasets

# Filter datasets based on arguments
if args.datasets is not None:
    # Map dataset names to tuples for cc_bench_datasets
    cc_bench_dict = {name: (group, name) for group, name in cc_bench_datasets}
    datasets = []
    for ds_name in args.datasets:
        if ds_name in cc_bench_dict:
            datasets.append(cc_bench_dict[ds_name])
        elif ds_name in feature_datasets:
            datasets.append(ds_name)
        else:
            raise ValueError(f"Dataset {ds_name} not recognized.")
else:
    datasets = all_datasets

# Define available methods
all_methods = ['GNN', 'LinkGNN', 'kwikcluster', 'modified_pivot', 'EMBOnly',' LinkEMBOnly', 'Linear', 'LinearLink']
if args.methods is not None:
    methods = args.methods
else:
    methods = all_methods

costs_data = []
objective = 'MatrixFactorization'


for dataset in  datasets:

    if dataset in cc_bench_datasets:
        data = load_suitesparse_dataset(root='../data', group=dataset[0], name=dataset[1], device='cpu')
        dataset = dataset[1]
    elif dataset in feature_datasets:
        if dataset in ['Cora', 'PubMed', 'CiteSeer']:
            data = load_Planetoid_dataset(root='../data', dataset=dataset, device='cpu')
        elif dataset in ['AmazonPhoto', 'AmazonComputers']:
            data = load_Amazon_dataset(root='../data', dataset=dataset, device='cpu')
        elif dataset == 'WikiCS':
            data = load_wikics_dataset(root='../data', device='cpu')
        elif dataset == 'GitHub':
            data = load_github_dataset(root='../data', device='cpu')
        elif dataset == 'FacebookPagePage':
            data = load_facebook_dataset(root='../data', device='cpu')
        elif dataset in ['OGBN-Arxiv']:
            data = load_OGBN_dataset(root='../data', dataset=dataset, device='cpu')
    else:
        raise ValueError(f"Dataset {dataset} not recognized.")

    feature_matrix = getattr(data, 'x', None)

    # Set max_clusters based on args or default behavior
    if args.max_clusters is not None:
        max_clusters = args.max_clusters
    else:
        max_clusters = 10000 if data.num_nodes > 10000 else data.num_nodes 
    print('Loaded dataset:', dataset, 'num_nodes:', data.num_nodes, 'num_edges:', data.num_edges)
    for method in methods:
        if method in ['GNN', 'LinkGNN', 'Linear', 'LinearLink']:
            in_channels = args.in_channels
            data.x = torch.nn.Embedding(data.num_nodes, in_channels, _freeze=True).weight
        else:
            in_channels = None
            data.x = torch.arange(data.num_nodes, device=device)
        if method == 'EMBOnly' or method == 'LinkEmbOnly':
            num_nodes = data.num_nodes

        if method in ['kwikcluster', 'modified_pivot']:
            for i in range(args.n_iter):
                seed_everything(i*1000 + 6199)
                cost, runtime, best_clustering = method_switchcase(method, data.clone(), device=device, max_clusters=max_clusters)

                curr_results = {
                    'num_nodes': data.num_nodes, 
                    'num_edges': data.num_edges,
                    'name': dataset,
                    'method': method,
                    'max_clusters': data.num_nodes,
                    'found_clusters': torch.unique(best_clustering).size(0),
                    'objective': 'MinDisagree',
                    'random_pivots': False,
                    'in_channels': None,
                    'patience': None,
                    'cost': cost, 
                    'runtime': runtime
                    }
                
                if hasattr(data, 'y') and data.y is not None:
                    labels = data.y.cpu().numpy().reshape(-1)
                    preds = best_clustering.cpu().numpy().reshape(-1)
                    nmi = normalized_mutual_info_score(labels, preds)
                    accuracy, f1_score = eval_pairwise_accuracy(labels, preds)
                    curr_results['NMI'] = nmi
                    curr_results['pairwise_accuracy'] = accuracy
                    curr_results['pairwise_f1'] = f1_score

                print(f"Method: {method}, Cost: {cost}, Runtime: {runtime:.2f} seconds")
                costs_data.append(curr_results)
                # save temporary results
                costs_data_df = pd.DataFrame(costs_data)
                costs_data_df.to_csv(f'../results/costs_real_data_{timestamp}.csv', index=False)
                torch.cuda.empty_cache()
        elif method in ['GNN', 'EMBOnly', 'MLP', 'EMBMLP', 'Linear']:
            for i in range(args.n_iter):
                seed_everything(i*1000 + 6199)
                cost, runtime, best_clustering = method_switchcase(method, data.clone(), objective=objective, 
                                                                    lr=args.lr, wd=args.weight_decay, 
                                                                    device=device, max_clusters=max_clusters, 
                                                                    random_pivots=args.random_pivots, in_channels=in_channels, 
                                                                    patience=args.patience, num_epochs=args.num_epochs)

                found_clusters = torch.unique(best_clustering).size(0)
                curr_results = {
                    'num_nodes': data.num_nodes, 
                    'num_edges': data.num_edges,
                    'name': dataset,
                    'method': method, 
                    'objective': objective,
                    'lr': args.lr,
                    'weight_decay': args.weight_decay,
                    'max_clusters': max_clusters,
                    'found_clusters': found_clusters,
                    'random_pivots': args.random_pivots,
                    'patience': args.patience,
                    'in_channels': in_channels,
                    'cost': cost, 
                    'runtime': runtime
                }
                if hasattr(data, 'y') and data.y is not None:
                    labels = data.y.cpu().numpy().reshape(-1)
                    preds = best_clustering.cpu().numpy().reshape(-1)
                    nmi = normalized_mutual_info_score(labels, preds)
                    accuracy, f1_score = eval_pairwise_accuracy(labels, preds)
                    curr_results['NMI'] = nmi
                    curr_results['pairwise_accuracy'] = accuracy
                    curr_results['pairwise_f1'] = f1_score

                costs_data.append(curr_results)
                print(f"Method: {method}, {objective}, Cost: {cost}, Runtime: {runtime:.2f} seconds")

                # save temporary results
                costs_data_df = pd.DataFrame(costs_data)
                costs_data_df.to_csv(f'../results/costs_real_data_{timestamp}.csv', index=False)
                torch.cuda.empty_cache()
        else:
            for i in range(args.n_iter):
                seed_everything(6199+i)
                hidden_channels = args.in_channels
                time_start = time.time()
                if method == 'LinearLink':
                    model = LinearModel(
                        in_channels=in_channels,
                        out_channels=hidden_channels,
                        ).to(device)
                elif method == 'LinkGNN':
                    model = GNNModel(in_channels=in_channels, hidden_channels=hidden_channels, bias=False).to(device)
                elif method == 'LinkEmbOnly':
                    model = EMBOnlyModel(
                        hidden_channels=hidden_channels,
                        num_nodes=data.num_nodes,
                        ).to(device)
                model = train_linkmodel(model, data, device, epochs=args.num_epochs, lr=args.lr, weight_decay=args.weight_decay, patience=args.patience, random_pivots=args.random_pivots)
                time_end = time.time()
                train_runtime = time_end - time_start

                cost = float('inf')
                runtime = train_runtime
                time_start = time.time()

                out = compute_out(model, data, device, batch_size=args.batch_size)
                time_end = time.time()
                runtime += time_end - time_start

                for threshold in np.arange(0.05, 1., 0.05,):
                    time_start = time.time()
                    pivot_clustering = make_cc_clusters(out, data.edge_index, threshold, device)
                    # pivot_clustering = make_pivot_clusters(out, device, threshold=threshold)
                    time_end = time.time()
                    runtime += time_end - time_start

                    pivot_cost = compute_cost_from_clustering_complete_graph(pivot_clustering, data.edge_index)
                    if pivot_cost < cost:
                        best_threshold = threshold
                        cost = pivot_cost
                        best_pivot_clustering = pivot_clustering

                if (best_pivot_clustering == -1).any():
                    raise ValueError('Some nodes were not assigned to any cluster.')
                found_clusters = torch.unique(best_pivot_clustering).size(0)
                curr_results = {
                    'num_nodes': data.num_nodes, 
                    'num_edges': data.num_edges,
                    'name': dataset,
                    'method': method,
                    'max_clusters': None,
                    'found_clusters': found_clusters,
                    'best_threshold': best_threshold,
                    'objective': 'LinkObjective',
                    'lr': args.lr,
                    'weight_decay': args.weight_decay,
                    'random_pivots': args.random_pivots,
                    'in_channels': in_channels,
                    'patience': args.patience,
                    'cost': cost, 
                    'train_runtime': train_runtime,
                    'runtime': runtime
                    }
                if hasattr(data, 'y') and data.y is not None:
                    labels = data.y.cpu().numpy().reshape(-1)
                    preds = best_pivot_clustering.cpu().numpy().reshape(-1)
                    nmi = normalized_mutual_info_score(labels, preds)
                    accuracy, f1_score = eval_pairwise_accuracy(labels, preds)
                    curr_results['NMI'] = nmi
                    curr_results['pairwise_accuracy'] = accuracy
                    curr_results['pairwise_f1'] = f1_score
                costs_data.append(curr_results)
                print(f"Method: {method}, Cost: {cost}, Runtime: {runtime:.2f} seconds")
                # save temporary results
                costs_data_df = pd.DataFrame(costs_data)
                costs_data_df.to_csv(f'../results/costs_real_data_{timestamp}.csv', index=False)
                torch.cuda.empty_cache()

