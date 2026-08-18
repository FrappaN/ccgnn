
import random
import subprocess  # FIX: used by get_gpu_memory_usage(), was never imported
import time        # FIX: used by wait_for_cuda_memory(), was never imported

import numpy as np
import torch
from sklearn.metrics.cluster import pair_confusion_matrix, normalized_mutual_info_score

def convert_matrix_to_clusters(cluster_assignment_matrix):
	# convert the clusters assignment matrix to a set of sets
	clusters = []
	nonzero_columns = cluster_assignment_matrix.nonzero()[:, 1].unique()
	for col in nonzero_columns:
		clusters.append(set(cluster_assignment_matrix[:, col].nonzero()[:, 0].tolist()))
		clusters = list(clusters)

	return clusters

def eval_pairwise_accuracy(labels, clusters):
	# compute pair confusion matrix
	pcm = pair_confusion_matrix(labels, clusters)
	tp = pcm[1, 1]
	fp = pcm[0, 1]
	fn = pcm[1, 0]
	tn = pcm[0, 0]
	accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
	f1_score = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
	return accuracy, f1_score




def evaluate_clustering(data, clustering):
	ground_truth = data.dataset_1['CID'].values.astype(int)
	num_points = ground_truth.shape[0]
	if len(clustering) < num_points:
		cluster_labels = np.zeros(num_points, dtype=int) - 1
		for cluster_id, cluster in enumerate(clustering):
			cluster_labels[np.array(list(cluster), dtype=int)] = cluster_id
	else:
		cluster_labels = np.array(clustering, dtype=int)
	
	nmi = normalized_mutual_info_score(ground_truth, cluster_labels)
	accuracy, f1_score = eval_pairwise_accuracy(ground_truth, cluster_labels)
	results_df = {
		'nmi': nmi,
		'pairwise_accuracy': accuracy,
		'pairwise_f1_score': f1_score
	}
	return results_df



def get_gpu_memory_usage():
    """
    Returns a dictionary of GPU memory usage by device.
    Keys are device IDs, values are tuples of (used_gib, total_gib).
    """
    try:
        # Run nvidia-smi command to get the GPU stats in a parsable format
        command = "nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits"
        result = subprocess.run(command.split(), capture_output=True, text=True, check=True)
        
        gpu_stats = {}
        # Parse the output line by line
        for line in result.stdout.strip().split('\n'):
            device_id, used_mib, total_mib = line.split(',')
            # Convert MiB to GiB for consistency
            used_gib = float(used_mib) / 1024
            total_gib = float(total_mib) / 1024
            gpu_stats[int(device_id)] = (used_gib, total_gib)
            
        return gpu_stats
        
    except FileNotFoundError:
        print("nvidia-smi not found. Make sure the NVIDIA driver is installed and in your PATH.")
        return {}
    except subprocess.CalledProcessError as e:
        print(f"Error running nvidia-smi: {e}")
        return {}
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return {}


def wait_for_cuda_memory(required_memory_gib, device, interval_minutes=10):
    """
    Waits until enough free memory is available on a specified CUDA device.

    Args:
        required_memory_gib (float): The amount of memory in GiB required to proceed.
        device_id (int): The ID of the CUDA device to check. Defaults to 0.
        interval_minutes (int): The time in minutes to wait between checks. Defaults to 10.
    """
    
    if not torch.cuda.is_available():
        print("CUDA is not available. The script will proceed on the CPU.")
        return

    while True:
        try:
            gpu_stats = get_gpu_memory_usage()
            total_memory_gib = gpu_stats[0][1]
            allocated_memory_gib = gpu_stats[0][0]

            # Calculate the free memory
            free_memory_gib = total_memory_gib - allocated_memory_gib
            
            print(f"Current free memory on CUDA device {device}: {free_memory_gib:.2f} GiB")
            
            # Check if enough memory is available
            if free_memory_gib >= required_memory_gib:
                print(f"Sufficient memory ({required_memory_gib} GiB) is now available. The script will proceed.")
                break
            else:
                print(f"Not enough memory. Waiting for {interval_minutes} minutes...")
                time.sleep(interval_minutes * 60)
        
        except RuntimeError as e:
            print(f"An error occurred while checking memory: {e}. Retrying in {interval_minutes} minutes...")
            time.sleep(interval_minutes * 60)



def compute_cost_from_clustering_complete_graph(clustering, edge_index):
	"""
	Number of disagreements of `clustering` under the complete signed-graph
	formulation of Eq. 2: cut edges + non-adjacent pairs placed together.
	Assumes edge_index stores both directions and has no self-loops.
	"""
	clustering = clustering.cpu()
	edge_index = edge_index.cpu()
	directed_edge_index = edge_index[:, edge_index[0] < edge_index[1]]
	clusters, counts = torch.unique(clustering, return_counts=True)
	diff_edge_mask = (clustering[directed_edge_index[0]] != clustering[directed_edge_index[1]]).float()
	pos_cost = torch.sum(diff_edge_mask).item()
	# now compute negative cost as the number of pairs of nodes in the same cluster not connected by an edge
	connected_pairs = directed_edge_index.size(1)-pos_cost
	total_pairs = torch.sum(counts*(counts-1)//2).item()
	#total_pairs = sum((counts[i]*(counts[i]-1))//2 for i in range(len(clusters)))
	neg_cost = total_pairs - connected_pairs
	cost = pos_cost + neg_cost
	return cost