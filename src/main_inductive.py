import os
os.environ['OMP_NUM_THREADS'] = '2'  # Limit the number of threads for parallel processing
os.environ['MKL_NUM_THREADS'] = '2'  # Limit the number of threads for MKL operations
os.environ['NUMEXPR_NUM_THREADS'] = '2'  # Limit the number of threads for NumExpr
os.environ['OPENBLAS_NUM_THREADS'] = '2'  # Limit the number of threads for OpenBLAS

import torch
import torch_geometric as tg
import pandas as pd
import numpy as np
import time
import subprocess
import argparse
from sklearn.metrics import normalized_mutual_info_score

from torch_geometric import seed_everything
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph



from methods import method_switchcase

from load_utils import load_Planetoid_dataset, load_Amazon_dataset, load_OGBN_dataset, \
                          load_wikics_dataset, load_github_dataset, load_facebook_dataset
from models import GNNModel, LinearModel
from train import train_linkmodel
from utils import compute_cost_from_clustering_complete_graph, eval_pairwise_accuracy
from train import train_linkmodel, make_cc_clusters, compute_out

from sklearn.metrics import normalized_mutual_info_score
# clusters_costs = {}
# cluster_optimal_costs = {}

# get day 
timestamp = time.strftime("%Y%m%d-%H%M%S")


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def inductive_train_split(data, train_ratio=0.8):
    num_nodes = data.num_nodes
    perm = torch.randperm(num_nodes)
    train_size = int(train_ratio * num_nodes)
    train_idx = perm[:train_size]
    test_idx = perm[train_size:]

    train_x = data.x[train_idx] if data.x is not None else None
    train_labels = data.y[train_idx] if data.y is not None else None
    train_edge_index, train_edge_attr = tg.utils.subgraph(train_idx, data.edge_index, data.edge_attr, relabel_nodes=True, num_nodes=num_nodes)
    train_data = tg.data.Data(x=train_x, edge_index=train_edge_index, edge_attr=train_edge_attr, y=train_labels)

    test_x = data.x[test_idx] if data.x is not None else None
    test_labels = data.y[test_idx] if data.y is not None else None
    test_edge_index, test_edge_attr = tg.utils.subgraph(test_idx, data.edge_index, data.edge_attr, relabel_nodes=True, num_nodes=num_nodes)
    test_data = tg.data.Data(x=test_x, edge_index=test_edge_index, edge_attr=test_edge_attr, y=test_labels)

    return train_data, test_data



# Parse command-line arguments
parser = argparse.ArgumentParser(description='Run inductive correlation clustering experiments on graph datasets.')
parser.add_argument('--n_iter', type=int, default=5, help='Number of iterations (default: 5)')
parser.add_argument('--patience', type=int, default=100, help='Patience for early stopping (default: 100)')
parser.add_argument('--random_pivots', type=int, default=1000, help='Number of random pivots (default: 1000)')
parser.add_argument('--max_clusters', type=int, default=None, help='Maximum number of clusters. If None, will be set to min(10000, num_nodes) (default: None)')
parser.add_argument('--in_channels', type=int, default=512, help='Number of input channels for embeddings (default: 512)')
parser.add_argument('--num_epochs', type=int, default=5000, help='Number of training epochs (default: 5000)')
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate (default: 0.01)')
parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay (default: 0.0)')
parser.add_argument('--batch_size', type=int, default=5000, help='Batch size for inference (default: 5000)')
parser.add_argument('--datasets', type=str, nargs='+', default=None, 
                    help='List of datasets to run. Options: Cora, PubMed, CiteSeer, AmazonPhoto, AmazonComputers, WikiCS, GitHub, FacebookPagePage, OGBN-Arxiv. If None, all datasets will be run (default: None)')
parser.add_argument('--methods', type=str, nargs='+', default=None,
                    help='List of methods to use. Options: GNN, LinkGNN, kwikcluster, modified_pivot, EMBOnly, LinkEMBOnly, Linear, LinearLink. If None, all methods will be used (default: None)')

args = parser.parse_args()

#datasets = [ ]  #('SNAP', 'loc-Brightkite'),

# Define available datasets
all_feature_datasets = ['Cora', 'PubMed', 'CiteSeer',  'AmazonPhoto', 'AmazonComputers', 'WikiCS', 'GitHub',  'FacebookPagePage', 'OGBN-Arxiv',] #'Cora', 'PubMed', 'CiteSeer',  'AmazonPhoto', 'AmazonComputers', 'WikiCS', 'GitHub',  'FacebookPagePage', 

# Filter datasets based on arguments
if args.datasets is not None:
    feature_datasets = args.datasets
else:
    feature_datasets = all_feature_datasets

# Define available methods
all_methods = ['GNN', 'LinkGNN', 'kwikcluster', 'modified_pivot', 'EMBOnly',' LinkEMBOnly', 'Linear', 'LinearLink']
if args.methods is not None:
    methods = args.methods
else:
    methods = all_methods

costs_data = []
objective = 'MatrixFactorization'


for dataset in feature_datasets:

    if dataset in ['Cora', 'PubMed', 'CiteSeer']:
        data = load_Planetoid_dataset(root='../data', dataset=dataset, device='cpu')
    elif dataset in ['AmazonPhoto', 'AmazonComputers']:
        data = load_Amazon_dataset(root='../data', dataset=dataset, device='cpu')
    elif dataset in ['OGBN-Arxiv']:
        data = load_OGBN_dataset(root='../data', dataset=dataset, device='cpu')
    elif dataset == 'WikiCS':
        data = load_wikics_dataset(root='../data', device='cpu')
    elif dataset == 'GitHub':
        data = load_github_dataset(root='../data', device='cpu')
    elif dataset == 'FacebookPagePage':
        data = load_facebook_dataset(root='../data', device='cpu')
    else:
        raise ValueError(f"Dataset {dataset} not recognized.")
    
    feature_matrix = getattr(data, 'x', None)

    # Set max_clusters based on args or default behavior
    if args.max_clusters is not None:
        max_clusters = args.max_clusters
    else:
        max_clusters = 10000 if data.num_nodes > 10000 else data.num_nodes 
    print('Loaded dataset:', dataset, 'max_clusters:', max_clusters)
    for method in methods:
        if method in ['GNN', 'LinkGNN', 'Linear', 'LinearLink']:
            in_channels = feature_matrix.size(1)
            data.x = feature_matrix

            lr = args.lr
            weight_decay = args.weight_decay
            
        else:
            in_channels = args.in_channels
            data.x = torch.arange(data.num_nodes, device=device)
            num_nodes = data.num_nodes
            num_clusters = max_clusters
            lr = 0.1
            weight_decay = 0.0


        if method in ['GNN', 'EMBOnly', 'MLP', 'EMBMLP', 'Linear']:
            for i in range(args.n_iter):
                seed_everything(i*1000 + 6199)
                train_graph, test_graph = inductive_train_split(data, train_ratio=0.5)

                _, runtime, _, model = method_switchcase(method, train_graph.clone(), objective=objective, 
                                                                    lr=lr, wd=weight_decay,
                                                                    device=device, max_clusters=max_clusters, 
                                                                    random_pivots=args.random_pivots, in_channels=in_channels, 
                                                                    patience=args.patience, return_model=True, num_epochs=args.num_epochs)

                # retest on full data
                time_start = time.time()
                best_out = compute_out(model, test_graph, device, batch_size=args.batch_size)
                cluster_assignment = torch.argmax(best_out, dim=1)
                time_end = time.time()
                inference_time = time_end - time_start
                runtime += time_end - time_start
                num_clusters = torch.unique(cluster_assignment).size(0)
                cost = compute_cost_from_clustering_complete_graph(cluster_assignment, test_graph.edge_index)

                curr_results = {
                    'num_nodes': data.num_nodes, 
                    'num_edges': data.num_edges,
                    'name': dataset,
                    'method': method, 
                    'objective': objective,
                    'lr': lr,
                    'weight_decay': weight_decay,
                    'max_clusters': max_clusters,
                    'random_pivots': args.random_pivots,
                    'patience': args.patience,
                    'in_channels': in_channels,
                    'runtime': runtime,
                    'inference_time': inference_time,
                    'test_cost': cost,
                    'found_clusters': num_clusters

                }

                if hasattr(test_graph, 'y') and test_graph.y is not None:
                    nmi = normalized_mutual_info_score(test_graph.y.cpu().numpy().reshape(-1), cluster_assignment.cpu().numpy().reshape(-1))
                    print(f'NMI: {nmi}')
                    accuracy, f1_score = eval_pairwise_accuracy(test_graph.y.cpu().numpy().reshape(-1), cluster_assignment.cpu().numpy().reshape(-1))
                    print(f'Accuracy: {accuracy}, F1-score: {f1_score}')
                    curr_results['NMI'] = nmi
                    curr_results['pairwise_accuracy'] = accuracy
                    curr_results['pairwise_f1'] = f1_score

                costs_data.append(curr_results)
                print(f"Method: {method}, {objective}, Cost: {cost}, Runtime: {runtime:.2f} seconds")

                # save temporary results
                costs_data_df = pd.DataFrame(costs_data)
                costs_data_df.to_csv(f'../results/costs_inductive_{timestamp}.csv', index=False)
                torch.cuda.empty_cache()
        elif method in ['LinkGNN', 'LinearLink']:
            for i in range(args.n_iter):
                seed_everything(6199+i)
                train_graph, test_graph = inductive_train_split(data, train_ratio=0.5)
                hidden_channels = 512
                time_start = time.time()
                if method == 'LinkGNN':
                    model = GNNModel(in_channels=in_channels, hidden_channels=hidden_channels, bias=False).to(device)
                elif method == 'LinearLink':
                    model = LinearModel(in_channels=in_channels, out_channels=hidden_channels).to(device)
                model = train_linkmodel(model, train_graph, device, epochs=args.num_epochs, lr=lr, weight_decay=weight_decay, patience=args.patience, random_pivots=args.random_pivots)


                # first select best threshold on train graph
                best_threshold = None
                best_train_clustering = None
                for threshold in np.arange(0.01, 1.0, 0.01):
                    train_out = compute_out(model, train_graph, device, batch_size=args.batch_size)
                    # pivot_clustering = make_pivot_clusters(train_out, device, threshold=threshold)
                    pivot_clustering = make_cc_clusters(train_out, train_graph.edge_index, threshold, device)
                    pivot_cost = compute_cost_from_clustering_complete_graph(pivot_clustering, train_graph.edge_index)
                    if best_threshold is None or pivot_cost < compute_cost_from_clustering_complete_graph(best_train_clustering, train_graph.edge_index):
                        best_threshold = threshold
                        best_train_clustering = pivot_clustering

                time_end = time.time()
                runtime = time_end - time_start

                cost = float('inf')
                time_start = time.time()
                out = compute_out(model, test_graph, device, batch_size=args.batch_size)
                best_pivot_clustering = make_cc_clusters(out, test_graph.edge_index, best_threshold, device)
                time_end = time.time()
                runtime += time_end - time_start
                inference_time = time_end - time_start
                cost = compute_cost_from_clustering_complete_graph(best_pivot_clustering.cpu(), test_graph.edge_index.cpu())

                
                best_num_clusters = torch.unique(best_pivot_clustering).size(0)
                curr_results = {
                    'num_nodes': data.num_nodes, 
                    'num_edges': data.num_edges,
                    'name': dataset,
                    'method': method,
                    'max_clusters': None,
                    'objective': 'LinkObjective',
                    'lr': lr,
                    'weight_decay': weight_decay,
                    'random_pivots': args.random_pivots,
                    'in_channels': in_channels,
                    'patience': args.patience,
                    'test_cost': cost, 
                    'found_clusters': best_num_clusters,
                    'best_threshold': best_threshold,
                    'runtime': runtime,
                    'inference_time': inference_time
                    }
                if hasattr(test_graph, 'y') and test_graph.y is not None:
                    nmi = normalized_mutual_info_score(test_graph.y.cpu().numpy().reshape(-1), best_pivot_clustering.cpu().numpy().reshape(-1))
                    accuracy, f1_score = eval_pairwise_accuracy(test_graph.y.cpu().numpy().reshape(-1), best_pivot_clustering.cpu().numpy().reshape(-1))
                    curr_results['NMI'] = nmi
                    curr_results['pairwise_accuracy'] = accuracy
                    curr_results['pairwise_f1'] = f1_score
                costs_data.append(curr_results)
                print(f"Method: {method}, Cost: {cost}, Runtime: {runtime:.2f} seconds")
                # save temporary results
                costs_data_df = pd.DataFrame(costs_data)
                costs_data_df.to_csv(f'../results/costs_inductive_{timestamp}.csv', index=False)
                torch.cuda.empty_cache()
        else:
            for i in range(args.n_iter):
                seed_everything(i*1000 + 6199)
                train_graph, test_graph = inductive_train_split(data, train_ratio=0.5)
                cost, runtime, best_clustering = method_switchcase(method, test_graph.clone(), device=device, max_clusters=max_clusters)

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
                    'test_cost': cost, 
                    'runtime': runtime
                    }
                
                if hasattr(test_graph, 'y') and test_graph.y is not None:
                    nmi = normalized_mutual_info_score(test_graph.y.cpu().numpy().reshape(-1), best_clustering.cpu().numpy().reshape(-1))
                    accuracy, f1_score = eval_pairwise_accuracy(test_graph.y.cpu().numpy().reshape(-1), best_clustering.cpu().numpy().reshape(-1))
                    curr_results['NMI'] = nmi
                    curr_results['pairwise_accuracy'] = accuracy
                    curr_results['pairwise_f1'] = f1_score

                print(f"Method: {method}, Cost: {cost}, Runtime: {runtime:.2f} seconds")
                costs_data.append(curr_results)
                # save temporary results
                costs_data_df = pd.DataFrame(costs_data)
                costs_data_df.to_csv(f'../results/costs_inductive_{timestamp}.csv', index=False)
                torch.cuda.empty_cache()


