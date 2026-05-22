import os
os.environ['OMP_NUM_THREADS'] = '2'  # Limit the number of threads for parallel processing
os.environ['MKL_NUM_THREADS'] = '2'  # Limit the number of threads for MKL operations
os.environ['NUMEXPR_NUM_THREADS'] = '2'  # Limit the number of threads for NumExpr
os.environ['OPENBLAS_NUM_THREADS'] = '2'  # Limit the number of threads for OpenBLAS

import torch
import pandas as pd
import numpy as np
import time

from torch_geometric import seed_everything


from methods import init_loss
from train import train_linkmodel, make_cc_clusters, compute_out, train_nodemodel
from utils import compute_cost_from_clustering_complete_graph
from models import GNNModel
from load_utils import load_suitesparse_dataset


import datetime
day = datetime.datetime.now().strftime("%Y-%m-%d")


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



n_iters = 5
costs_data = []


hyperparams_GNN = {
    'patience': [100,],
    'random_pivots': [ 1000,],
    'in_channels': [512,],
    'num_layers': [1, 2,3],
    'conv_type': ['GCN', 'GraphSAGE', 'GIN', 'GAT'],
    'lr': [0.01,],
    'wd': [0.,],
}

hyperparams_LinkGNN = {
    'random_pivots': [1000,],
    'patience': [100,],
    'in_channels': [512,],
    'lr': [0.01,],
    'wd': [0,],
    'num_layers': [1, 2, 3],
    'conv_type': ['GCN', 'GraphSAGE', 'GIN', 'GAT'],
}

results_data = []
cc_bench_datasets = [('Newman', 'polblogs'), ('SNAP', 'ca-GrQc'), ('SNAP', 'ca-HepTh'), ('SNAP', 'ca-AstroPh'), ('SNAP', 'email-Enron'), ('Newman', 'cond-mat-2005')]

for dataset in cc_bench_datasets:
    dataset_name = dataset[1]
    data = load_suitesparse_dataset(root='../data', group=dataset[0], name=dataset[1], device='cpu')
    data.W = None
    objective = 'MatrixFactorization'

    num_edges = data.num_edges
    num_nodes = data.num_nodes
    max_clusters = 10000 if num_nodes > 10000 else num_nodes


        
    method_name = 'GNNModel'
    # test each hyperparameter combination for GNN


    for in_channels in hyperparams_GNN['in_channels']:
        data.x = torch.nn.Embedding(data.num_nodes, in_channels, _freeze=True).weight
        for random_pivots in hyperparams_GNN['random_pivots']:
            for patience in hyperparams_GNN['patience']:
                if max_clusters is None:
                    max_clusters = num_nodes
                for lr in hyperparams_GNN['lr']:
                    for wd in hyperparams_GNN['wd']:
                        for conv_type in hyperparams_GNN['conv_type']:
                            for num_layers in hyperparams_GNN['num_layers']:
                                for i in range(n_iters):
                                    try:
                                        seed_everything(6199+i*2599)

                                        time_start = time.time()
                                        loss = init_loss(objective=objective)
                                        model = GNNModel(in_channels=in_channels, hidden_channels=in_channels, out_channels=max_clusters, num_layers=num_layers, conv_type=conv_type).to(device)

                                        out = train_nodemodel(model, data, loss, lr=lr, wd=wd, num_epochs=5000, patience=patience, random_pivots=random_pivots, device=device)
                                        clustering = out.argmax(dim=1)

                                        cost = compute_cost_from_clustering_complete_graph(clustering.to(device), data.edge_index.to(device))

                                        time_end = time.time()
                                        runtime = time_end - time_start
                                        results_data.append({
                                            'name': dataset_name,
                                            'method': method_name, 
                                            'max_clusters': max_clusters,
                                            'patience': patience,
                                            'random_pivots': random_pivots,
                                            'in_channels': in_channels,
                                            'lr': lr,
                                            'wd': wd,
                                            'num_layers': num_layers,
                                            'conv_type': conv_type,
                                            'cost': cost, 
                                            'runtime': runtime,
                                        })
                                        results_df = pd.DataFrame(results_data)
                                        results_df.to_csv(f'../results/hyperparams_analysis_{day}_2.csv', index=False)
                                        torch.cuda.empty_cache()
                                    except Exception as e:
                                        results_data.append({
                                            'name': dataset_name,
                                            'method': method_name, 
                                            'max_clusters': max_clusters,
                                            'patience': patience,
                                            'random_pivots': random_pivots,
                                            'in_channels': in_channels,
                                            'lr': lr,
                                            'wd': wd,
                                            'num_layers': num_layers,
                                            'conv_type': conv_type,
                                            'cost': None, 
                                            'runtime': None,
                                            'error': str(e),
                                        })
                                        results_df  = pd.DataFrame(results_data)
                                        results_df.to_csv(f'../results/hyperparams_analysis_{day}_2.csv', index=False)
                                        torch.cuda.empty_cache()
                                        

    method_name = 'LinkGNNModel'
    # test each hyperparameter combination for LinkGNN

    for in_channels in hyperparams_LinkGNN['in_channels']:
        data.x = torch.nn.Embedding(data.num_nodes, in_channels, _freeze=True).weight
        for random_pivots in hyperparams_LinkGNN['random_pivots']:
            for patience in hyperparams_LinkGNN['patience']:
                if max_clusters is None:
                    max_clusters = num_nodes
                for lr in hyperparams_LinkGNN['lr']:
                    for wd in hyperparams_LinkGNN['wd']:
                        for conv_type in hyperparams_LinkGNN['conv_type']:
                            for num_layers in hyperparams_LinkGNN['num_layers']:
                                for i in range(n_iters):
                                    try:
                                        seed_everything(6199+i*2599)

                                        time_start = time.time()
                                        model = GNNModel(in_channels=in_channels, hidden_channels=in_channels, bias=False, conv_type=conv_type, num_layers=num_layers).to(device)
                                        model = train_linkmodel(model, data, device, epochs=5000, lr=lr, weight_decay=wd, patience=patience, random_pivots=random_pivots)
                                        time_end = time.time()
                                        train_runtime = time_end - time_start

                                        cost = float('inf')
                                        runtime = train_runtime
                                        time_start = time.time()

                                        out = compute_out(model, data, device, batch_size=5000)
                                        time_end = time.time()
                                        runtime += time_end - time_start

                                        for threshold in np.arange(0.05, 1., 0.05,):
                                            time_start = time.time()
                                            pivot_clustering = make_cc_clusters(out.to(device), data.edge_index, threshold, device)
                                            # pivot_clustering = make_pivot_clusters(out, device, threshold=threshold)
                                            time_end = time.time()
                                            runtime += time_end - time_start

                                            pivot_cost = compute_cost_from_clustering_complete_graph(pivot_clustering.to(device), data.edge_index.to(device))
                                            if pivot_cost < cost:
                                                best_threshold = threshold
                                                cost = pivot_cost
                                                best_pivot_clustering = pivot_clustering

                                        results_data.append({
                                            'name': dataset_name,
                                            'method': method_name, 
                                            'max_clusters': max_clusters,
                                            'patience': patience,
                                            'random_pivots': random_pivots,
                                            'in_channels': in_channels,
                                            'lr': lr,
                                            'wd': wd,
                                            'num_layers': num_layers,
                                            'conv_type': conv_type,
                                            'cost': cost, 
                                            'runtime': runtime,
                                        })
                                        results_df = pd.DataFrame(results_data)
                                        results_df.to_csv(f'../results/hyperparams_analysis_{day}.csv', index=False)
                                        torch.cuda.empty_cache()
                                    except Exception as e:
                                        results_data.append({
                                            'name': dataset_name,
                                            'method': method_name, 
                                            'max_clusters': max_clusters,
                                            'patience': patience,
                                            'random_pivots': random_pivots,
                                            'in_channels': in_channels,
                                            'lr': lr,
                                            'wd': wd,
                                            'num_layers': num_layers,
                                            'conv_type': conv_type,
                                            'cost': None, 
                                            'runtime': None,
                                            'error': str(e),
                                        })
                                        torch.cuda.empty_cache()
