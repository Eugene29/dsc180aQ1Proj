import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch_geometric import utils
import networkx as nx
from scipy.sparse.linalg import eigsh
from torch_geometric.utils import to_dense_adj
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix, to_undirected, to_dense_adj, scatter
from torch_geometric.utils.num_nodes import maybe_num_nodes
from scipy.sparse.linalg import eigsh
import cupy as cp

def train_graphs(m, optimizer, train_loader, val_loader, args, accelerator):
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    # Assume a model, optimizer, data_loader and scheduler are defined
    m, optim, train_loader, val_loader = accelerator.prepare(m, optimizer, train_loader, val_loader)
    
    for epoch in range(args["epochs"]):
        total_loss = 0
        correct = 0
        total = 0
        m.train()
        for data in train_loader:
#             data = data.to(dev)
            optimizer.zero_grad(set_to_none=True)
            out = m(data)
            loss = F.cross_entropy(out, data.y)
            accelerator.backward(loss)
            optimizer.step()
            total_loss += loss.item() 

            pred_prob = F.softmax(out, dim=-1)
            pred = torch.argmax(pred_prob, dim=-1)
            if len(pred.shape) < len(data.y.shape):
                data.y = torch.argmax(data.y, dim=-1)
#             sparse_y = torch.argmax(data.y, dim=-1)
            correct += (pred == data.y).sum().item() # this would be tensor
            total += data.batch[-1].item()+1 ## get the last batch_id and +1 becuz id starts from 0
        total_loss /= len(train_loader)
        train_acc = correct / total


        if epoch % 10 == 0 or epoch == args["epochs"] - 1:
            m.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                val_loss = 0
                for data in val_loader:
#                     data = data.to(dev)
                    out = m(data)
                    loss = F.cross_entropy(out, data.y)

                    pred_prob = F.softmax(out, dim=-1)
                    pred = torch.argmax(pred_prob, dim=-1)
#                     sparse_y = torch.argmax(data.y, dim=-1)
                    if len(pred.shape) < len(data.y.shape):
                        data.y = torch.argmax(data.y, dim=-1)
                    correct += (pred == data.y).sum().item() # this would be tensor
                    total += data.batch[-1].item()+1 ## get the last batch_id and +1 becuz id starts from 0
                    val_loss += loss.item()
            val_loss /= len(val_loader)
            val_acc = correct / total
            
            train_losses += [total_loss]
            val_losses += [val_loss]
            train_accs += [train_acc]
            val_accs += [val_acc]
            accelerator.print(f"Epoch {epoch}:\t total_loss: {total_loss:.4f}\t train_acc: {train_acc:.4f}\t  val_loss: {val_loss:.4f}\t val_acc: {val_acc:.4f}")
    return train_losses, val_losses, train_accs, val_accs


def train_one_graph(m, optimizer, data, args, accelerator):
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    # Assume a model, optimizer, data_loader and scheduler are defined
#     m, optim, data.x, data.edge_index, data.y = accelerator.prepare(m, optimizer, data.x, data.edge_index, data.y)
    m, optim, data = accelerator.prepare(m, optimizer, data)
#     data = data.to(accelerator.device)
    
#     accelerator.print(data.x.device)
#     accelerator.print(data.x.shape)
    accelerator.print("train_one_graph function")
    data = next(iter(data))
    for epoch in range(args["epochs"]):
        total_loss = 0
        correct = 0
        m.train()
        ## move to cuda
#         data.x = data.x.to(dev)
#         data.y = data.y.to(dev)
#         data.edge_index = data.edge_index#.to(dev)
#         data.train_mask = data.train_mask.to(dev)
#         data.val_mask = data.train_mask.to(dev)
        optimizer.zero_grad(set_to_none=True)
        out = m(data)
        out = out[data.train_mask]
        
        tlabel = data.y[data.train_mask]
        loss = F.cross_entropy(out, tlabel)
        accelerator.backward(loss)
#         loss.backward()
        optimizer.step()
        total_loss += loss.item() 

        pred_prob = F.softmax(out, dim=-1)
        pred = torch.argmax(pred_prob, dim=-1)
#         sparse_y = torch.argmax(tlabel, dim=-1)
        correct += (pred == tlabel).sum().item() # this would be tensor
        total = pred.shape[0]
#         total_loss /= 
#         print("below is sparse_y")
#         print(sparse_y)
#         print(torch.unique(sparse_y, return_counts=True))
        train_acc = correct / total

        if epoch % 5 == 0 or epoch == args["epochs"] - 1:
            m.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                val_loss = 0
                out = m(data)
                vlabel = data.y
#                 if args["mask"] is not None:
                out = out[data.val_mask]
                vlabel = vlabel[data.val_mask]
                loss = F.cross_entropy(out, vlabel)

                pred_prob = F.softmax(out, dim=-1)
                pred = torch.argmax(pred_prob, dim=-1)
#                 sparse_y = torch.argmax(vlabel, dim=-1)
                correct += (pred == vlabel).sum().item() # this would be tensor
                total += pred.shape[0]
                val_loss += loss.item()
#             val_loss /= total
            val_acc = correct / total
            
            train_losses += [total_loss]
            val_losses += [val_loss]
            train_accs += [train_acc]
            val_accs += [val_acc]
            accelerator.print(f"Epoch {epoch}:\t total_loss: {total_loss:.4f}\t train_acc: {train_acc:.4f}\t  val_loss: {val_loss:.4f}\t val_acc: {val_acc:.4f}")
    return train_losses, val_losses, train_accs, val_accs


def get_mask(dataset, train_size=0.7, val_size=0.3, test_size=0, accelerator=None):
    ## get_mask is for one_graph tasks.
    ## sets the train, val, test mask for dataset.
    
    num_label = dataset.data.y.shape[0]
    train_size = int(num_label * train_size)
    val_size = int(num_label * val_size)
    
    train_mask = torch.zeros(num_label)
    train_mask[:train_size] = 1
    val_mask = torch.zeros(num_label)
    val_mask[train_size: train_size + val_size] = 1
    test_mask = torch.zeros(num_label)
    test_mask[train_size + val_size:] = 1
    accelerator.print(f"Data: {dataset.data}")
    accelerator.print(f"Total labels: {num_label}")
    try:
        accelerator.print(f"Num of classes: {dataset.data.num_classes}")
    except:
        accelerator.print(f"Num of classes: {dataset.data.y.max().item() + 1}")
    accelerator.print(f"Num of feature dimensions: {dataset.data.num_features}")
    accelerator.print(f"Distribution of labels: {torch.unique(dataset.data.y, return_counts=True)}")
    dataset.data.train_mask = train_mask.to(torch.bool)
    dataset.data.val_mask = val_mask.to(torch.bool)
    dataset.data.test_mask = test_mask.to(torch.bool)
    

# from scipy.sparse.linalg import eigsh
# def compute_laplacian(graph, k, N):
#     edge_index = graph.edge_index
#     lap_edges, lap_weights = utils.get_laplacian(edge_index, normalization="sym")
#     L = to_dense_adj(edge_index=lap_edges, edge_attr=lap_weights, max_num_nodes=N).squeeze()#[48, 0]
# #         eigval, eigvec  = torch.linalg.eigh(L)
#     eigval, eigvec = eigsh(L.numpy(), k=k, ncv=40)
#     lapse = eigvec[:, :k]
#     lapse = torch.tensor(eigvec[:, :k])
#     depth = lapse.size(1)
#     if depth != k:
#         lapse = torch.concat([lapse, torch.zeros((N, k - depth))], dim=-1)
#     return lapse

# def feature_augmentation(datasetlst, features, k=2):
#     ## features=True if it already has features.
#     for graph in datasetlst:
#         N = graph.x.shape[0] if features else graph.num_nodes
#         concat_edges = torch.cat([graph.edge_index[0], graph.edge_index[1]], dim=-1)
#         degrees = utils.degree(concat_edges, num_nodes=N).view(-1, 1) ## degrees
#         constant = torch.ones((N, 1)) ## constants
#         lapse = compute_laplacian(graph, k, N=N)

# #         print(eigval)
        
#         nx_graph = utils.to_networkx(graph)
#         clustering = nx.clustering(nx_graph)
#         clustering = [v for k, v in clustering.items()]
#         clustering = torch.tensor(clustering).view(-1, 1) ## constants
#         if features:
#             graph.x = torch.cat([graph.x, constant, degrees, clustering, lapse], dim=-1)
#         else:
#             graph.x = torch.cat([constant, degrees, clustering, lapse], dim=-1)

def compute_clustering_coefficient(edge_index, max_num_nodes):
    adj = to_dense_adj(edge_index, max_num_nodes=max_num_nodes).squeeze(0)
    deg = adj.sum(dim=1)
    triangle = torch.mm(adj, torch.mm(adj, adj))
    clustering = triangle.diag() / (deg * (deg - 1))
    # Handling NaN values for nodes with degree 1 or 0
    clustering[deg <= 1] = 0.0
    return clustering
# from scipy.sparse.linalg import eigsh
def compute_laplacian(graph, k, N, dev):
    edge_i, edge_w = get_laplacian(graph.edge_index, normalization="sym", num_nodes=N)
    dense_L = to_dense_adj(edge_index=edge_i, edge_attr=edge_w).squeeze()
    L = cp.asarray(dense_L)
    eigval, eigvec = cp.linalg.eigh(L)
    lapse = torch.tensor(eigvec[:, :k], device=dev)
    length = lapse.size(1)
    if length < k: ## zero padding if k > length
        lapse = torch.concat([lapse, torch.zeros((N, k - length), device=dev)], dim=-1)
    return lapse

def feature_augmentation(datasetlst, features, dev, k=4):
    ## features=True if it already has features.
    for graph in datasetlst:
        if graph.x is not None:
            graph.x = graph.x.to(dev)
        graph.edge_index = graph.edge_index.to(dev)
        num_nodes = graph.x.shape[0] if features else graph.num_nodes
        concat_edges = torch.cat([graph.edge_index[0], graph.edge_index[1]], dim=-1)
        degrees = utils.degree(concat_edges, num_nodes=num_nodes).view(-1, 1) ## degrees
        constant = torch.ones((num_nodes, 1), device=dev) ## constants
        clustering = compute_clustering_coefficient(graph.edge_index, num_nodes).view(-1, 1) # clustering
#         print(degrees.device, constant.device, clustering.device, graph.edge_index.device)
        lapse = compute_laplacian(graph, k=k, N=num_nodes, dev=dev) ## Lapse
#         RWSE = get_rw_landing_probs(ksteps=range(1, 17), edge_index=graph.edge_index, num_nodes=graph.x.size(0))
        if features:
#             cpu = torch.device("cpu")
            graph.x = torch.cat([graph.x, constant, degrees, clustering, lapse], dim=-1).to(dev)#.to(cpu)
#             graph.x = torch.cat([graph.x, constant, degrees, clustering], dim=-1).to(dev)#.to(cpu)
#             print(torch.cat([constant, degrees, clustering, lapse], dim=-1).shape)
        else:
            graph.x = torch.cat([constant, degrees, clustering,], dim=-1).to(dev)
        if hasattr(graph, "pe"):
            graph.x = torch.cat([graph.x, graph.pe.to(dev)], dim=-1)
    
def prepare_data(dataset, og_dataset=None, train_size=0.7, val_size=0.3, test_size=0, accelerator=None):
    ## for multiple graphs.
    ## returns train, val, test split graphs.
    train_size = int(len(dataset) * train_size)
    val_size = int(len(dataset) * val_size)
    train = dataset[:train_size]
    val = dataset[train_size:train_size + val_size]
    test = dataset[train_size + val_size:]
    if og_dataset is not None:
        dataset = og_dataset
    accelerator.print(f"Data: {dataset}")
    accelerator.print(f"length of dataset: {len(dataset)}")
    accelerator.print(f"Total labels: {dataset.data.y.shape[0]}")
    accelerator.print(f"Num of classes: {dataset.num_classes}")
    accelerator.print(f"Num of feature dimensions: {dataset.num_features}")
    accelerator.print(f"Distribution of labels: {torch.unique(dataset.data.y, return_counts=True)}")
    return train, val, test


def visualize(train_losses, val_losses, train_accs, val_accs):
    plt.plot(train_losses, label="train_loss")
    plt.plot(val_losses, label="val_loss")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend()
    plt.show()
    plt.plot(train_accs, label="train_acc")
    plt.plot(val_accs, label="val_acc")
    plt.xlabel("epochs")
    plt.ylabel("acc")
    plt.legend()
    plt.show()