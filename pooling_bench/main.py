import argparse
from argparse import RawTextHelpFormatter
import numpy as np
import torch
import time
import torch.nn.functional as F
import pandas as pd
from torch_geometric.loader import DataLoader
from torch_geometric import seed_everything
from torch_geometric.datasets import TUDataset
from torch_geometric.transforms import Constant, Compose

from scripts.nn_model import GIN_Pool_Net
from scripts.utils import DataToFloat, log
from tgp.poolers import pooler_map
from tgp.datasets import MultipartiteGraphDataset, EXPWL1Dataset

from sklearn.metrics import roc_auc_score


try:
    from ogb.graphproppred import PygGraphPropPredDataset
except Exception:
    PygGraphPropPredDataset = None

seed_everything(6199)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
parser.add_argument('--linear_cc', action='store_true', default=False, help='Use linear layer for CCPool instead of GCN')
parser.add_argument('--pool_ratio', type=float, default=0.1)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--hidden_channels', type=int, default=64)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--max_epochs', type=int, default=1000)
parser.add_argument('--runs', type=int, default=10)
parser.add_argument('--patience', type=int, default=100)
parser.add_argument('--results_path', type=str, default='./results/')
parser.add_argument('--verbose', action='store_true', default=False, help='Print stuff for debugging')
parser.add_argument('--dataset', type=str, default='EXPWL1', choices=['EXPWL1', 'MUTAG', 'NCI1', 'REDDIT-BINARY', 'ogbg-molhiv', 'multipartite'], help='Dataset to use')
args = parser.parse_args()
print(args)

timestamp = time.strftime("%Y%m%d-%H%M%S")

def train(model, loader, optimizer, loss_fn):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out, aux_loss = model(data, verbose=args.verbose)
        loss = loss_fn(out, data.y) + aux_loss
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * data.num_graphs
    return total_loss / len(loader.dataset)

@torch.no_grad()
def test(model, loader, loss_fn):
    model.eval()
    total_correct = 0
    total_loss = 0
    preds = []
    labels = []
    for data in loader:
        data = data.to(device)
        out, _ = model(data, verbose=args.verbose)
        loss = loss_fn(out, data.y)
        total_loss += float(loss) * data.num_graphs
        pred = out.argmax(dim=-1)
        total_correct += int((pred == data.y).sum())
        preds.append(out)
        labels.append(data.y)

    if args.dataset == 'ogbg-molhiv':
        preds = torch.cat(preds, dim=0).cpu()
        labels = torch.cat(labels, dim=0).cpu()
        # Compute ROC-AUC
        try:
            rocauc = roc_auc_score(labels.numpy(), preds[:, 1].numpy())
            return rocauc, total_loss / len(loader.dataset)
        except Exception:
            pass
    return total_correct / len(loader.dataset), total_loss / len(loader.dataset)

pooling_methods =  [ 'ccpool', 'nopool', 'jb', 'hosc', 'dmon', 'acc', 'kmis', 'maxcut',]
all_results = []
for pooling in pooling_methods:
    try:
        ### Dataset
        if args.dataset == 'EXPWL1':
            path = "data/EXPWL1/"
            dataset = EXPWL1Dataset(path, transform=DataToFloat())
            pre_transform = None
        elif args.dataset == 'multipartite':
            path = "data/multipartite"
            dataset = MultipartiteGraphDataset(path)
            pre_transform = None
        else:
            # Get TGP pooler class if available to fetch data transforms
            pre_transform = None
            if pooling in pooler_map:
                pooler_cls = pooler_map[pooling]
                try:
                    pre_transform = pooler_cls.data_transforms()
                except Exception:
                    pre_transform = None

            if args.dataset == 'REDDIT-BINARY':
                if pre_transform is None:
                    pre_transform = Constant()
                else:
                    transform = Constant()
                    pre_transform = Compose([pre_transform, transform])

            if args.dataset in ['MUTAG', 'NCI1', 'REDDIT-BINARY']:
                dataset = TUDataset(root='data/TUDataset', name=args.dataset, pre_transform=pre_transform, use_node_attr=True, force_reload=False)
            elif args.dataset == 'ogbg-molhiv':
                if PygGraphPropPredDataset is None:
                    raise RuntimeError("ogb is not installed. Please install ogb to use ogbg-molhiv.")
                dataset = PygGraphPropPredDataset(name='ogbg-molhiv', root='data/OGB', pre_transform=pre_transform)
                # Filter out missing labels (-1)
                keep_idx = []
                for i, d in enumerate(dataset):
                    y = d.y.view(-1).tolist()
                    val = y[0] if len(y) > 0 else -1
                    if val in [0, 1]:
                        keep_idx.append(i)
                dataset = dataset[keep_idx]
                dataset.data.y = dataset.data.y.view(-1)

                
            else:
                raise ValueError(f"Unsupported dataset: {args.dataset}")

        max_nodes = 0
        total_nodes = 0
        total_edges = 0
        for d in dataset:
            total_nodes += d.num_nodes
            total_edges += d.num_edges
            max_nodes = max(d.num_nodes, max_nodes)
        avg_nodes = max(1, total_nodes // len(dataset))
        avg_edges = max(1, total_edges // len(dataset))

        num_features = dataset.num_features
        if num_features is None:
            num_features = dataset[0].x.shape[1]

        num_classes = getattr(dataset, 'num_classes', None)
        if num_classes is None:
            num_classes = torch.max(dataset[0].y).item() + 1

        loss_fn = F.nll_loss

        if pooling in ['sparse-random', 'ccpool', 'ccpool_v2']:
            max_nodes *= args.batch_size


        pool_str = pooling
        if pooling == 'None':
            pooling = None
            pool_str = 'No-Pool'
        
        num_pivots = None
        # if args.dataset == 'REDDIT-BINARY':
        #     num_pivots = 100*args.batch_size

        k_order = None
        if pooling == 'kmis':
            if args.dataset == 'EXPWL1':
                k_order = 5
            elif args.dataset == 'MUTAG':
                k_order = 6
            elif args.dataset == 'NCI1':
                k_order = 7
            elif args.dataset == 'ogbg-molhiv':
                k_order = 7
            elif args.dataset == 'REDDIT-BINARY':
                k_order = 4
            elif args.dataset == 'multipartite':
                k_order = 2

            
        pr = args.pool_ratio
        if pool_str == 'ccpool':
            if args.dataset == 'EXPWL1':
                pr = 0.1
            elif args.dataset == 'MUTAG':
                pr = 0.05
            elif args.dataset == 'NCI1':
                pr = 0.075
            elif args.dataset == 'ogbg-molhiv':
                pr = 0.065
            elif args.dataset == 'REDDIT-BINARY':
                pr = 0.11
            elif args.dataset == 'multipartite':
                pr = 0.21

            

        ### Training
        tot_acc = []
        tot_times = []

        rnd_idx = np.random.permutation(len(dataset))
        dataset = dataset[list(rnd_idx)]

        # create ten-fold splits
        # at each iteration, do train on 80%, val on 10%, test on 10%
        fold_size = len(dataset) // 10
        folds = [np.arange(i * fold_size, (i + 1) * fold_size) for i in range(9)]
        folds.append(np.arange(9 * fold_size, len(dataset)))
        train_splits = []
        val_splits = []
        test_splits = []
        for i in range(10):
            if args.dataset == 'ogbg-molhiv' and pooling == 'acc' and i < 7:
                # already did 7 runs for acc on ogbg-molhiv
                continue
                
            test_idx = folds[i]
            val_idx = folds[(i + 1) % 10]
            train_idx = np.hstack([folds[j] for j in range(10) if j != i and j != (i + 1) % 10])
            train_splits.append(train_idx)
            val_splits.append(val_idx)
            test_splits.append(test_idx)


        for r in range(args.runs): 

            train_dataset = dataset[train_splits[r]]
            val_dataset = dataset[val_splits[r]]
            test_dataset = dataset[test_splits[r]]
            
            train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, args.batch_size)
            test_loader = DataLoader(test_dataset, args.batch_size)
            # Choose model: TGP poolers or legacy
            use_tgp = (pooling in pooler_map) or (pooling == 'ccpool') or (pooling == 'ccpool_v2')
            out_channels = getattr(train_dataset, 'num_classes', None)
            if out_channels is None:
                # e.g., ogbg-molhiv is binary
                out_channels = 2

            net_model = GIN_Pool_Net(
                in_channels=train_dataset.num_features,
                out_channels=out_channels,
                hidden_channels=args.hidden_channels,
                pooling=pooling,
                pool_ratio=pr,
                average_nodes=avg_nodes,
                linear_cc=args.linear_cc,
                max_nodes=max_nodes,
                k_order=k_order,
                num_pivots=num_pivots,
            ).to(device)
            opt = torch.optim.Adam(net_model.parameters(), lr=args.lr)    
            
            best_val = np.inf
            best_test = 0
            total_training_time = 0
            patience_counter = 0

            if pooling == 'ccpool_v2':
                # Pre-train CCPool pooler
                if args.verbose:
                    print("Pre-training CCPool pooler...")
                net_model.pooler.train_pooler(train_loader)
                if args.verbose:
                    print("Pre-training done.")

            for epoch in range(1, args.max_epochs + 1):
                start = time.time()
                loss = train(net_model, train_loader, opt, loss_fn)
                end = time.time()
                tot_time = end - start
                total_training_time += tot_time
                train_acc, _ = test(net_model, train_loader, loss_fn)
                val_acc, val_loss = test(net_model, val_loader, loss_fn)
                test_acc, _ = test(net_model, test_loader, loss_fn)
                
                if val_loss < best_val:
                    best_val = val_loss
                    best_test = test_acc
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= args.patience:
                        #print(f"Early stopping at epoch {epoch} due to no improvement in validation loss.")
                        break
                
                log(Epoch=epoch, Loss=loss, Val_Loss=val_loss, Train=train_acc, Val=val_acc, Test=test_acc, Time=tot_time)
                
            avg_epoch_time = total_training_time / epoch
            # copy params to results
            curr_results = {}
            for k, v in vars(args).items():
                if k == 'results_path':
                    continue
            curr_results['pooling'] = pool_str
            curr_results['dataset'] = args.dataset
            curr_results['run'] = r
            curr_results['hidden_channels'] = args.hidden_channels
            curr_results['batch_size'] = args.batch_size
            curr_results['max_epochs'] = args.max_epochs
            curr_results['patience'] = args.patience
            curr_results['lr'] = args.lr
            curr_results['pool_ratio'] = pr
            if args.dataset == 'ogbg-molhiv':
                curr_results['best_test_rocauc'] = best_test
            else:
                curr_results['best_test_acc'] = best_test
            curr_results['total_epochs'] = epoch
            curr_results['total_training_time'] = total_training_time
            curr_results['avg_epoch_time'] = avg_epoch_time

            all_results.append(curr_results)
            # Save all results to a single CSV
            all_results_df = pd.DataFrame(all_results)
            all_results_df.to_csv(f'{args.results_path}all_results_{args.dataset}_{timestamp}.csv', index=False)
            print("Results saved to CSV.")
            del net_model, opt, train_loader, val_loader, test_loader
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"Error with pooling method {pooling}: {e}")
        results = {}
        results['dataset'] = args.dataset
        results['pooling'] = pooling
        results['error'] = str(e)
        all_results.append(results)
        # Save all results to a single CSV
        all_results_df = pd.DataFrame(all_results)
        all_results_df.to_csv(f'{args.results_path}all_results_{args.dataset}_{timestamp}.csv', index=False)
        torch.cuda.empty_cache()
        print("Results saved to CSV.")