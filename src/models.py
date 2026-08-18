import torch
import torch_geometric as tg
import torch_geometric.utils as tgu
from scipy import sparse as sp

def laplacian_p(edge_index, num_nodes, pos_enc_dim):
    """
        Graph positional encoding v/ Laplacian eigenvectors
    """

    # Laplacian

    A = tgu.to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes).astype(float)
    degrees = tgu.degree(edge_index[0], num_nodes=num_nodes).numpy()

    N = sp.diags(degrees ** -0.5, dtype=float)
    L = sp.eye(num_nodes) - N * A * N

    # Eigenvectors with scipy
    #EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim+1, which='SR')
    EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim+1, which='SR', tol=1e-2) # for 40 PEs
    EigVec = EigVec[:, EigVal.argsort()] # increasing order
    
    return torch.from_numpy(EigVec[:,1:pos_enc_dim+1]).float()


class GNNModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels=None, bias=False, num_layers=1, conv_type='GCN', trainable_emb=False, input_linearity=False):
        super(GNNModel, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.input_linearity = input_linearity
        if input_linearity:
            self.input_linear = torch.nn.Linear(in_channels, hidden_channels)

        for _ in range(num_layers - 1):
            if conv_type == 'GCN':
                self.convs.append(tg.nn.GCNConv(in_channels, hidden_channels, bias=bias))
            elif conv_type == 'GraphSAGE':
                self.convs.append(tg.nn.SAGEConv(in_channels, hidden_channels, bias=bias, aggr='sum'))
            elif conv_type == 'GAT':
                self.convs.append(tg.nn.GATConv(in_channels, hidden_channels, bias=bias))
            elif conv_type == 'GIN':
                nn = tg.nn.MLP([in_channels, hidden_channels, hidden_channels])
                self.convs.append(tg.nn.GINConv(nn))

        if out_channels is None:
            out_channels = hidden_channels
        if num_layers == 1 and not input_linearity:
            hidden_channels = in_channels
        if conv_type == 'GCN':
            self.conv = tg.nn.GCNConv(hidden_channels, out_channels, bias=bias)
        elif conv_type == 'GraphSAGE':
            self.conv = tg.nn.SAGEConv(hidden_channels, out_channels, bias=bias, aggr='sum')
        elif conv_type == 'GAT':
            self.conv = tg.nn.GATConv(hidden_channels, out_channels, bias=bias)
        elif conv_type == 'GIN':
            nn = tg.nn.MLP([hidden_channels, hidden_channels, out_channels])
            self.conv = tg.nn.GINConv(nn)
        else:
            raise ValueError(f"Convolution type {conv_type} not recognized.")
        self.convs.append(self.conv)
        self.num_features = hidden_channels

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        if self.input_linearity:
            x = self.input_linear(x)
        for conv in self.convs[:-1]:
            x = conv(x, edge_index).relu()
        x = self.convs[-1](x, edge_index)
        return x 



class LinearModel(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(in_channels, out_channels)

    def forward(self, data):
        x = data.x
        x = self.linear(x)
        return x

class MLPModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_nodes, num_layers=4, no_gnn_train=True):
        super(MLPModel, self).__init__()
        self.gnn = GNNModel(in_channels, hidden_channels, num_nodes)
        # freeze the GNN layers
        if no_gnn_train:
            for param in self.gnn.parameters():
                param.requires_grad = False
        self.layers = torch.nn.ModuleList()
        for _ in range(num_layers - 1):
            self.layers.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.fc_out = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x = self.gnn(data).relu()
        for layer in self.layers:
            x = layer(x).relu()
        x = self.fc_out(x)
        return x

class EMBMLPModel(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_nodes, num_layers=4):
        super(EMBMLPModel, self).__init__()
        self.embedding = torch.nn.Embedding(num_nodes, hidden_channels)
        self.layers = torch.nn.ModuleList()
        for _ in range(num_layers - 1):
            self.layers.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.fc_out = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x = self.embedding(data.x).relu()
        for layer in self.layers:
            x = layer(x).relu()
        x = self.fc_out(x)
        return x

class EMBOnlyModel(torch.nn.Module):
    def __init__(self, num_nodes, hidden_channels=32):
        super(EMBOnlyModel, self).__init__()
        self.embedding = torch.nn.Embedding(num_nodes, hidden_channels)

    def forward(self, data):
        x = self.embedding(data.x)
        return x