import torch 
import torch_geometric as pyg
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset, Planetoid, LRGBDataset#, CitationFull
import networkx as nx
from torch_geometric import utils
from utils import prepare_data, get_mask, train_one_graph, train_graphs, visualize, compute_laplacian, feature_augmentation
# from GNNs import GCN, GNN_one_graph, GATv2, GIN, GCN_VN, GIN_VN, GATv2_VN, GraphGPS
from GNNs import GNN, GCN_one_graph, GraphGPS, GATv2
from torch_geometric.utils import to_dense_adj
import time
import numpy as np
from accelerate import Accelerator
import sys
import torch_geometric.transforms as T

dataset_name = sys.argv[1]
conv = sys.argv[2]
vn = sys.argv[3] == "True"
one_graph = False
accelerator = Accelerator()

with accelerator.main_process_first():
    if dataset_name == "Cora":
        #### Cora ####
        one_graph = True
        dataset = Planetoid(root=f"/tmp/{dataset_name}", name=f"{dataset_name}")
        dataset = dataset.shuffle()
        
        args = {
            "epochs":40,
            "dropout":0.6,
            "one_graph": True,
            "mask":None,
            "lr":0.01,
            "num_hops":2,
            "heads":8,
            "graph_task": False
               }
        ## One_graph: sets train, val, test mask on the dataset
    #     feature_augmentation(datasetlst, features=True, dev=accelerator.device)
        get_mask(dataset, accelerator=accelerator)
        input_dim = dataset.x.size(1)
        hidden_dim = 64
        output_dim = dataset.num_classes
        dataset = DataLoader(dataset, batch_size=1, shuffle=True)
#         accelerator.print([x for x in dataset])
        

    elif dataset_name == "Enzymes":
        #### Enzymes ####
        dataset_name = "ENZYMES"
        transform = T.AddRandomWalkPE(walk_length=17, attr_name='pe')
    #     train_dataset = ZINC(path, subset=True, split='train', pre_transform=transform)
        dataset = TUDataset(root=f"/tmp/{dataset_name}", name=f"{dataset_name}", pre_transform=transform)

#         dataset = TUDataset(root=f"/tmp/{dataset_name}", name=f"{dataset_name}")
        dataset = dataset.shuffle()
        datasetlst = list(dataset)
        args = {
            "epochs":750,
            "batch_size":32,
            "dropout":0.2,
            "one_graph": False,
            "mask":None,
            "lr":0.0001,
            "heads":5,
            "num_hops": 2,
            "graph_task": True
               }
    #     feature_augmentation(datasetlst, features=True)
        feature_augmentation(datasetlst, features=True, dev=accelerator.device)
#         accelerator.print([data.shape in data in datasetlst])

        ## Use datasetlst when feature augmenting
        train, val, test = prepare_data(datasetlst, dataset, accelerator=accelerator) 
        batch_size = args["batch_size"]
        train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val, batch_size=batch_size)
        test_loader = DataLoader(test, batch_size=batch_size)
        ## Build Model
        input_dim = datasetlst[0].x.size(1) ## 3 augmented
        hidden_dim = 64 # 5 * input_dim
        output_dim = dataset.num_classes
        print(input_dim)

    elif dataset_name == "IMDB":
        ###IMDB####
        dataset_name = "IMDB-BINARY"
        transform = T.AddRandomWalkPE(walk_length=17, attr_name='pe')
        dataset = TUDataset(root=f"/tmp/{dataset_name}", name=f"{dataset_name}", pre_transform=transform)
        dataset = dataset.shuffle()
        datasetlst = list(dataset)
        feature_augmentation(datasetlst, features=False, dev=accelerator.device)
        args = {
            "epochs":500,
            "batch_size":32,
            "dropout":0.2,
            "one_graph": False,
            "mask":None,
            "lr":0.005,
            "heads":3,
            "num_hops":2,
            "graph_task": True
               }
        ## Use datasetlst when feature augmenting
        train, val, test = prepare_data(datasetlst, dataset, accelerator=accelerator) 
        batch_size = args["batch_size"]
        train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val, batch_size=batch_size)
        test_loader = DataLoader(test, batch_size=batch_size)
        ## Build Model
        input_dim = datasetlst[0].x.size(1) #dataset.num_features + 3
        hidden_dim = 32
        output_dim = dataset.num_classes
        accelerator.print(hidden_dim)
        accelerator.print(input_dim)

    elif dataset_name == "Peptides-func":
        ###IMDB####
        transform = T.AddRandomWalkPE(walk_length=17, attr_name='pe')
    #     dataset_name = "IMDB-BINARY"
    #     dataset_name = "Peptides-struct"
        dataset_name = "Peptides-func"
        dataset = LRGBDataset(root=f"/tmp/{dataset_name}", name=f"{dataset_name}", pre_transform=transform)
        dataset = dataset.shuffle()
        datasetlst = list(dataset)
        feature_augmentation(datasetlst, features=True, dev=accelerator.device)
        args = {
            "epochs":200,
            "batch_size":64,
            "dropout":0.1,
            "one_graph": False,
            "mask":None,
#             "lr":0.0001,
            "lr":0.0005,
            "num_hops":4,
            "graph_task": True,
               }
        ## Use datasetlst when feature augmenting
        train, val, test = prepare_data(datasetlst, dataset, accelerator=accelerator) 
        batch_size = args["batch_size"]
        train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val, batch_size=batch_size)
        test_loader = DataLoader(test, batch_size=batch_size)
        ## Build Model
        input_dim = datasetlst[0].x.size(1) #dataset.num_features + 3
        hidden_dim = 120
        output_dim = dataset.num_classes
        accelerator.print(hidden_dim)
        accelerator.print(input_dim)
        accelerator.print(datasetlst[0])
    else:
        raise IndexError

#### Training ####
m = GNN(input_dim, hidden_dim, output_dim, conv=conv, dropout=args["dropout"], num_hops=args["num_hops"], vn=vn, args=args)
optim = torch.optim.AdamW(m.parameters(), lr = args["lr"], weight_decay = 5e-4)
if one_graph:
    start = time.time()
    train_losses, val_losses, train_accs, val_accs = train_one_graph(m, optim, dataset, args=args, accelerator=accelerator)
else:
    start = time.time()
    train_losses, val_losses, train_accs, val_accs = train_graphs(m, optim, train_loader, val_loader, args, accelerator=accelerator)
    
end = time.time()
accelerator.print(f"total time taken: {end - start}")
# idx = np.argmax(val_accs)
max_train = np.max(train_accs)
max_val = np.max(val_accs)
accelerator.print(f"best train: {max_train} \n best val: {max_val}")


