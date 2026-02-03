import os
import numpy as np
import torch
import torch_geometric as tg
import networkx as nx

from torch_geometric import seed_everything

from torch_geometric.datasets import Planetoid, Amazon, WikiCS, GitHub, FacebookPagePage
from ogb.nodeproppred import PygNodePropPredDataset


from torch_geometric.utils import to_undirected, remove_self_loops
from torch_geometric.transforms import Compose, ToUndirected, RemoveSelfLoops, RemoveIsolatedNodes, LargestConnectedComponents


from torch import Tensor




def generate_sbm_graph(num_nodes, num_communities, p_in, p_out, randomized_partitions=False):

    # various sizes of communities
    if randomized_partitions:
        sizes = np.random.uniform(0, 1, num_communities)
        sizes = (sizes / np.sum(sizes) * num_nodes).astype(int)
        sizes = np.maximum(sizes, 1)  # ensure at least one node in each community
    else:
        # equal sizes for each community
        sizes = [num_nodes // num_communities] * num_communities
    
    # distribute the remainder to the last community
    if np.sum(sizes) < num_nodes:
        sizes[-1] += num_nodes - np.sum(sizes)
    p = np.zeros((num_communities, num_communities))

    p.fill(p_out)
    np.fill_diagonal(p, p_in)
    graph = nx.stochastic_block_model(sizes, p, directed=False)
    return graph

def generate_sbm_dataset(num_graphs, num_nodes, p_in, p_out, num_partitions, seed=42, randomized_partitions=False):
    seed_everything(seed)
    graphs = []

    for i in range(num_graphs):
        num_partitions = num_partitions
        graph = generate_sbm_graph(num_nodes, num_partitions, p_in, p_out, randomized_partitions=randomized_partitions)
        graphs.append(graph)

    # convert to PyTorch tensors and add the weight matrix 
    pyg_graphs = convert_to_pyg(graphs)
    return pyg_graphs

def convert_to_pyg(graphs):
    pyg_graphs = []
    for i in range(len(graphs)):
        graph = graphs[i]
        pyg_graph = tg.utils.from_networkx(graph)
        # shuffle the node order
        permutation = torch.randperm(pyg_graph.num_nodes)
        row, col = pyg_graph.edge_index
        row = permutation[row]
        col = permutation[col]
        pyg_graph.edge_index = torch.vstack([row, col])
        pyg_graph.x = torch.arange(pyg_graph.num_nodes).view(-1)
        pyg_graph.edge_weight = None

        # create a weight matrix
        W = torch.ones((pyg_graph.num_nodes, pyg_graph.num_nodes))*(-1)

        W[pyg_graph.edge_index[0, :], pyg_graph.edge_index[1, :]] = 1
        # set the diagonal to 0
        for j in range(pyg_graph.num_nodes):
            W[j, j] = 0

        pyg_graph.W = W
        assignment_matrix = torch.zeros((pyg_graph.num_nodes, len(graph.graph['partition'])))
        for i in range(len(graph.graph['partition'])):
            assignment_matrix[permutation[np.array(list(graph.graph['partition'][i]))], i] = 1
        pyg_graph.partition = assignment_matrix
        pyg_graphs.append(pyg_graph.cpu())
        torch.cuda.empty_cache()
    return pyg_graphs


def load_Planetoid_dataset(root='data', dataset='Cora', device='cpu', LCC=False):
    dataset = Planetoid(root=root, name=dataset)
    data = dataset[0]

    if LCC:
        transform = LargestConnectedComponents()
        data = transform(data)

    transform = Compose([ToUndirected(), RemoveSelfLoops(), RemoveIsolatedNodes()])
    data = transform(data)

    data = data.to(device)
    data.num_pos_edges = data.num_edges
    return data

def load_Amazon_dataset(root='data', dataset='AmazonComputers', device='cpu', LCC=False):
    if dataset == 'AmazonComputers':
        dataset = Amazon(root=root, name='computers')
    elif dataset == 'AmazonPhoto':
        dataset = Amazon(root=root, name='photo')
    data = dataset[0]
    if LCC:
        transform = LargestConnectedComponents()
        data = transform(data)

    transform = Compose([ToUndirected(), RemoveSelfLoops(), RemoveIsolatedNodes()])
    data = transform(data)

    data = data.to(device)
    data.num_pos_edges = data.num_edges
    return data

def load_OGBN_dataset(root='data', dataset='OGBN-Arxiv', device='cpu', LCC=False):
    lowercase_name = dataset.lower()
    dataset = PygNodePropPredDataset(name=lowercase_name, root=root)
    data = dataset[0]

    if LCC:
        transform = LargestConnectedComponents()
        data = transform(data)
    transform = Compose([ToUndirected(), RemoveSelfLoops(), RemoveIsolatedNodes()])
    data = transform(data)

    data.num_pos_edges = data.num_edges

    data = data.to(device)
    return data


def load_wikics_dataset(root='data', device='cpu', LCC=False):
    dataset = WikiCS(root=root)
    data = dataset[0]
    if LCC:
        transform = LargestConnectedComponents()
        data = transform(data)

    transform = Compose([ToUndirected(), RemoveSelfLoops(), RemoveIsolatedNodes()])
    data = transform(data)

    data = data.to(device)
    data.num_pos_edges = data.num_edges
    return data

def load_github_dataset(root='data', device='cpu', LCC=False):
    root = os.path.join(root, 'github')
    dataset = GitHub(root=root)
    data = dataset[0]

    if LCC:
        transform = LargestConnectedComponents()
        data = transform(data)

    transform = Compose([ToUndirected(), RemoveSelfLoops(), RemoveIsolatedNodes()])
    data = transform(data)

    data = data.to(device)
    data.num_pos_edges = data.num_edges
    return data

def load_facebook_dataset(root='data', device='cpu', LCC=False):
    root = os.path.join(root, 'facebook')
    dataset = FacebookPagePage(root=root)
    data = dataset[0]
    if LCC:
        transform = LargestConnectedComponents()
        data = transform(data)

    transform = Compose([ToUndirected(), RemoveSelfLoops(), RemoveIsolatedNodes()])
    data = transform(data)

    data = data.to(device)
    data.num_pos_edges = data.num_edges
    return data


def load_suitesparse_dataset(root='data', group='SNAP', name='email-Enron', device='cpu'):
    # options: ('SNAP', 'loc-Brightkite'), ('SNAP', 'ca-AstroPh'), ('SNAP', 'email-Enron'),
    # ('SNAP', 'ca-GrQc'), ('SNAP', 'ca-HepTh'), ('Newman', 'cond-mat-2005'), ('Newman', 'polblogs'), 
    dataset = tg.datasets.SuiteSparseMatrixCollection(root=root, group=group, name=name)
    data = dataset[0]


    transform = Compose([ToUndirected(), RemoveSelfLoops(), RemoveIsolatedNodes()])
    data = transform(data)
    data = data.to(device)

    data.num_pos_edges = data.num_edges

    return data
