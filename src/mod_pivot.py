import random
from collections import defaultdict
from sortedcontainers import SortedDict
from collections import defaultdict
import scipy.io
import matplotlib.pyplot as plt
from itertools import combinations
import numpy as np
import networkx as nx
import torch
import copy

class Node:
    def __init__(self, id, rank):
        self.id = id
        self.rank = rank
        self.neighbors = set()
        self.eliminator_pivot = None  # Eliminator for pivot clustering
        self.eliminator_modified = None  # Eliminator for modified pivot clustering
        self.n_plus = SortedDict()  # Neighbors with higher rank, indexed by ID
        self.n_minus = SortedDict()  # Neighbors with lower rank, indexed by k(u)
        self.m_pivot = 0  # Binary variable for LFMIS in pivot
        self.m_modified = 0  # Binary variable for LFMIS in modified pivot

    def update_neighbors(self, graph):
        self.n_plus.clear()
        self.n_minus.clear()

        for neighbor in self.neighbors:
            neighbor_node = graph[neighbor]
            if neighbor_node.rank > self.rank:
                self.n_plus[neighbor] = neighbor_node.rank
            else:
                self.n_minus[neighbor_node.rank] = neighbor

class CorrelationClustering:
    def __init__(self, edge_index: torch.Tensor, seed:int=100):
        graph = convert_pyg_graph_to_dict(edge_index)  # Convert NetworkX graph to dictionary format

        lists = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        self.delta = 0.5
        self.epsilon = 0.5
        random.seed(seed)
        self.initial_ranks = {}
        # Create a sorted list of vertices to ensure consistent ordering
        self.sorted_vertices = sorted(graph.keys())
        for v in self.sorted_vertices:
            self.initial_ranks[v] = random.random()
        self.graph = {v: Node(v, self.initial_ranks[v]) for v in self.sorted_vertices}  # Create Node objects for each vertex
        for v, neighbors in graph.items():
            self.graph[v].neighbors = set(neighbors)


        self.pivots_modified = set()
        self.clusters_modified = defaultdict(set)  # Clusters for modified pivot algorithm
        self.disagreements_modified = 0  # Disagreements for modified pivot algorithm
        # Preprocessing step
        self._preprocess()

        best_clusters_modified = None
        self.delta_best = 0
        self.epsilon_best = 0
        min_cost = 10000000000
        for i in lists:
            for j in lists:
                self.delta = i
                self.epsilon = j
               
                self._compute_modified()

                self._update_disagreements()

                if min_cost > self.disagreements_modified:

                    min_cost = self.disagreements_modified
                    best_clusters_modified = copy.deepcopy(self.clusters_modified)
                    self.delta_best = self.delta

                    self.epsilon_best = self.epsilon
              
        self.disagreements_modified = min_cost
        self.clusters_modified = best_clusters_modified


    def _preprocess(self):
        """Perform preprocessing to initialize rankings and data structures."""
        for v in self.sorted_vertices:
            node = self.graph[v]
            node.m_pivot = 1  # Initially assume v is in LFMIS for pivot
            node.m_modified = 1  # Initially assume v is in LFMIS for modified pivot
            node.eliminator_pivot = v  # v is its own eliminator for pivot
            node.eliminator_modified = v  # v is its own eliminator for modified pivot

    def _compute_modified(self):


        self.pivots_modified.clear()
        self.clusters_modified.clear()
        alive = set(self.sorted_vertices)

        A = set()  # Initialize the set A
        k = 0
        self.pivots_list = []
        while alive:
            # Find vertex with minimum rank among alive vertices
            pivot = min(alive, key=lambda x: self.graph[x].rank)

            # if pivot in A:
            #     # add the pivot to singleton cluster as it will not be added later
            #     self.clusters_modified[pivot] = {pivot,}

            # Add v as a pivot and create its cluster
            self.pivots_modified.add(pivot)

            # Update eliminator and m_pivot values
            self.graph[pivot].eliminator_pivot = pivot
            self.graph[pivot].m_pivot = 1
            Cv = [pivot,]

            # Process v's neighbors
            for neighbor in self.graph[pivot].neighbors & alive:
                self.graph[neighbor].eliminator_pivot = pivot
                self.graph[neighbor].m_pivot = 0
                Cv.append(neighbor)

            # Convert Cv to a set for processing
            Cv = set(Cv)
            alive -= Cv  # Remove Cv from alive vertices

            # Define Dv as vertices in Cv with very different neighborhoods
            Dv = {u for u in Cv if len(self.graph[u].neighbors & Cv) <= self.delta * len(Cv) - 1}

            # Define Dv' as a subsample of Dv
            Dv_prime = set(random.sample(list(Dv), min(len(Dv), int(self.delta * len(Cv)))))

         # Define Av as vertices in the neighborhood of neighbors of v with neighborhoods similar to Cv
         # Only look at neighborhood of Cv to find the A set, instead of looking at the whole graph
            Av = {
                w
                for neighbor in Cv  # First-level neighbors of v
                for w in self.graph[neighbor].neighbors  # Second-level neighbors (neighbors of neighbors)
                if w not in Cv and w not in A and len(self.graph[w].neighbors ^ Cv) <= self.epsilon * len(Cv) - 1
            }

            # Define A'v as a subsample of Av
            Av_prime = set(random.sample(list(Av), min(len(Av), int(self.delta * len(Cv)))))

            # Step 9: Put vertices of (Dv' \ A) ∪ (Av \ A'v) in singleton clusters
            singleton_vertices = (Dv_prime - A) | (Av - Av_prime)
            for sv in singleton_vertices:
                self.clusters_modified[k] = {sv}
                k += 1
                self.pivots_list.append(sv)

            # Step 10: Put all vertices of (Cv ∪ A'v) \ (Dv' ∪ A) in the same cluster
            cluster_vertices = (Cv | Av_prime) - (Dv_prime | A)
            self.clusters_modified[k] = cluster_vertices
            k += 1
            self.pivots_list.append(pivot)

            # Update A to include Av
            A |= Av
        # Call duplicate removal after creating initial clusters
        self._remove_duplicate_vertices()
        # remove empty clusters
        self.clusters_modified = {j: cluster for j, cluster in self.clusters_modified.items() if cluster}
        

    def _remove_duplicate_vertices(self):
        # Create a mapping of vertex to its clusters
        vertex_to_clusters = defaultdict(list)
        for curr_k, cluster in self.clusters_modified.items():
            for vertex in cluster:
                vertex_to_clusters[vertex].append(curr_k)

        # For vertices in multiple clusters, remove from all but lowest pivot
        for vertex, ks in vertex_to_clusters.items():
            if len(ks) > 1:
                ranking_ks = [self.graph[self.pivots_list[k]].rank for k in ks]
                # Sort pivots to get consistent ordering based on rank
                sorted_ks = [k for _, k in sorted(zip(ranking_ks, ks))]
                # Keep vertex only in cluster with lowest pivot
                lowest_pivot = sorted_ks[0]
                # Remove from all other clusters
                for curr_k in sorted_ks[1:]:
                    self.clusters_modified[curr_k].discard(vertex)

    def _update_disagreements(self):
        self.disagreements_modified = self._calculate_disagreements(self.clusters_modified)

    def get_cluster_size(self):

        size_p = [len(cluster) for cluster in self.clusters_pivot.values() if cluster]
        size_m = [len(cluster) for cluster in self.clusters_modified.values() if cluster]


        return size_p, size_m




    def _calculate_disagreements(self, clusters):
        disagreements = 0

        # Process clusters in a consistent order
        sorted_clusters = sorted(clusters.items())

        for _, cluster_nodes in sorted_clusters:
            # Skip empty clusters
            if not cluster_nodes:
                continue

            nodes = sorted(cluster_nodes)  # Sort nodes for consistency
            cluster_size = len(nodes)
            # Calculate non-edge disagreements within the cluster
            total_pairs = (cluster_size * (cluster_size - 1) )// 2  # cluster_size choose 2
            internal_edges = 0
            external_edges = 0
            for i in nodes:
                for j in self.graph[i].neighbors:
                    if j in nodes:

                        internal_edges += 1
                    else:
                        external_edges += 1

            non_edge_disagreements = total_pairs - (internal_edges//2)
            disagreements += non_edge_disagreements


            disagreements += (external_edges//2)  # Count each external edge

        return disagreements


    def get_clusters(self):
        return self.clusters_modified

    def get_disagreements(self):
        return self.disagreements_modified


def convert_nx_graph_to_dict(G):
    """Convert a NetworkX graph to a dictionary representation."""
    graph_dict = {}
    for node in G.nodes():
        graph_dict[node] = list(G.neighbors(node))
    return graph_dict


def convert_pyg_graph_to_dict(edge_index):
    """Convert an edge_index to a dictionary representation."""
    directed_edge_index = edge_index[:, edge_index[0] > edge_index[1]] 
    num_nodes = directed_edge_index.max().item() + 1
    g_dict = {i: [] for i in range(num_nodes)}
    for edge in directed_edge_index.t().tolist():
        src, dst = edge
        g_dict[src].append(dst)
        g_dict[dst].append(src)
    return g_dict

# # Constructing the SBM  
# def generate_clustered_graph(num_nodes, num_clusters):
#     # Create a dictionary to represent the graph
#     G = {i: [] for i in range(num_nodes)}

#     # Assign nodes to clusters
#     clusters = {i: [] for i in range(num_clusters)}
#     node_cluster_map = {}
#     for node in range(num_nodes):
#         cluster = random.randint(0, num_clusters - 1)
#         clusters[cluster].append(node)
#         node_cluster_map[node] = cluster

#     # Iterate through all pairs of nodes
#     for node1, node2 in combinations(range(num_nodes), 2):
#         cluster1, cluster2 = node_cluster_map[node1], node_cluster_map[node2]

#         # Determine probability of edge based on clustering
#         if cluster1 == cluster2:
#             edge_probability = 0.9
#         else:
#             edge_probability = 0.1

#         # Add edge with specified probability
#         if random.random() < edge_probability:
#             G[node1].append(node2)
#             G[node2].append(node1)

#     return G, clusters

# # Running the SBM. 
# # Parameters
# num_nodes_list = [ 100,200,300]
# num_clusters_list = [5,6,7,8,9,10,20]
# k = 100

# # Lists to store average and worst cases
# avg_pivot_disagreements = []
# avg_modified_disagreements = []

# worst_disagreements = []

# for num_nodes in num_nodes_list:
#     for num_clusters in num_clusters_list:
#         sum_modified = 0
#         sum_pivot = 0
#         worst_case = float('inf')


#         # Generate the graph
#         G, clusters = generate_clustered_graph(num_nodes, num_clusters)

#         for _ in range(k):
#             cc = CorrelationClustering(G, random.random())
#             p, m = cc.get_disagreements()
#             sum_pivot += p
#             sum_modified += m
#             if m/p < worst_case:
#                 worst_case= m/p


#         avg_pivot_disagreements.append(sum_pivot / k)
#         avg_modified_disagreements.append(sum_modified / k)
#         worst_disagreements.append(worst_case)