# Inductive Correlation Clustering with Graph Neural Networks

This repository contains the official implementation for the paper: "Inductive Correlation Clustering with Graph Neural Networks".

The core experiments for Correlation Clustering presented in the paper can be replicated using the following scripts in the `src/` folder:

- `main.py`: Executes experiments in the transductive setting.
- `main_inductive.py`: Executes experiments in the inductive setting.
- `test_hyperparams.py`: Executes ablation experiments in the transductive setting.
- `ablation_ind.py`: Executes ablation experiment in the inductive setting.
- `threshold_sweep.py`: Executes the sensitivity analysis experiment in the inductive setting.

Running either script without arguments will execute all methods across all datasets using default hyperparameters. 
The `pooling_bench/` folder, instead, contains all code necessary for the replication of the pooling experiments.

## Replication
To replicate specific subsets of the experiments, use the --methods and --datasets arguments:

```
python main.py --methods GNN LinkGNN modified_pivot --datasets polblogs ca-GrQc Cora
python main_inductive.py --methods GNN LinkGNN modified_pivot --datasets MUTAG REDDIT-BINARY
```

## Pooling Benchmark

The `pooling_bench` directory contains code adapted from [The expressive power of pooling in Graph Neural Networks](https://github.com/FilippoMB/The-expressive-power-of-pooling-in-GNNs). To run the full pooling benchmark, navigate to the directory and use the --dataset flag for each of the datasets:

```
python main.py --dataset MUTAG
```
