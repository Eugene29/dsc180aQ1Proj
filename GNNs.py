import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as pyg
from torch_geometric.nn import Linear, GCNConv, global_mean_pool, global_add_pool, GATv2Conv, GINConv, GPSConv
from torch_geometric import utils

## Baseline GCN model. 
# class block_1(nn.module):
class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, conv, args, heads=8, dropout=0.1, num_hops=5, vn=False, ):
        super().__init__()        
        self.num_hops = num_hops
        self.dropout = dropout
        self.graphgps = conv == "GraphGPS"
        self.graph_task = args["graph_task"]
        h_dims = [hidden_dim for _ in range(num_hops)]

        def create_ffnn(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=hidden_dim, vn=False):
            if vn:
                ffnn = nn.Sequential(
                Linear(input_dim, 2 * hidden_dim),
                nn.BatchNorm1d(2 * hidden_dim),
                nn.GELU(),
                Linear(2 * hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
            )
            else:
                ffnn = nn.Sequential(
                Linear(input_dim, 2 * hidden_dim),
                nn.GELU(),
                Linear(2 * hidden_dim, output_dim),
                nn.GELU(),
            )
            return ffnn
        
        self.preprocess = create_ffnn(input_dim=input_dim)
        if vn:
            self.vn = nn.Embedding(1, hidden_dim)
            self.vn_mlps = nn.ModuleList([create_ffnn(vn=True) for _ in h_dims[:-1]])
        if conv == "GCN":
            self.convs = nn.ModuleList([GCNConv(h, h) for h in h_dims])
        elif conv == "GAT":
            self.convs = nn.ModuleList([GATv2Conv(h, h) for h in h_dims])
        elif conv == "GIN":
            self.convs = nn.ModuleList([GINConv(create_ffnn()) for h in h_dims])
        elif conv == "GraphGPS":
            self.convs = nn.ModuleList([GPSConv(h, GATv2Conv(h, h//heads, heads=heads), heads=4, dropout=dropout) for h in h_dims])
        else:
            raise IndexError
        if not self.graphgps: ## GPSConv comes with this
            self.ffnns = nn.ModuleList([create_ffnn() for h in h_dims])
            self.bns = nn.ModuleList(nn.BatchNorm1d(h) for h in h_dims)
        self.postprocess = create_ffnn(output_dim=output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        batch = getattr(data, "batch", None)
        x = x.float()
        
        x = self.preprocess(x)
        if hasattr(self, "vn"):
            if batch is None:
                vn = self.vn(0)
            else:
                vn = self.vn(torch.zeros(batch[-1].item() + 1, dtype=torch.long, device=x.device))
#         x = F.dropout(x, p=self.dropout)
        for i in range(self.num_hops):
            if hasattr(self, "vn"):
                if batch is None:
                    x = x + vn
                else:
                    x = x + vn[batch]
            if self.graphgps:
                x = self.convs[i](x, edge_index, batch=batch)
            else:
                x = self.convs[i](x, edge_index) + x
                x = self.bns[i](x)
    #             x = self.lns[i](x)
                x = self.ffnns[i](x) + x
                x = F.dropout(x, p=self.dropout)
            if hasattr(self, "vn") and i < self.num_hops - 1: ## ignore the last layer??
#                 vn = F.dropout(self.vn_mlps[i](global_add_pool(x, batch=batch)), p=0.2) + vn
#                 vn = vn + global_add_pool(x, batch=batch)
                vn = vn + global_mean_pool(x, batch=batch)
                vn = F.dropout(self.vn_mlps[i](vn), p=0)

        if self.graph_task:
            x = global_add_pool(x, batch)
        x = self.postprocess(x)
        return x
    
class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1, num_hops=5):
        super().__init__()
        self.num_hops = num_hops

        def create_ffnn(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=hidden_dim):
            ffnn = nn.Sequential(
                Linear(input_dim, hidden_dim),
#                 nn.ReLU(),
                nn.GELU(),
                Linear(hidden_dim, output_dim)
            )
            return ffnn
        

        self.preprocess = create_ffnn(input_dim=input_dim)
        self.ffnns = nn.ModuleList([create_ffnn() for _ in range(num_hops)]) # 1 for pre_process
        self.convs = nn.ModuleList([GCNConv(hidden_dim, hidden_dim) for _ in range(num_hops)])
        self.postprocess = create_ffnn(output_dim=output_dim)
        self.bns = nn.ModuleList(nn.BatchNorm1d(hidden_dim) for _ in range(num_hops))
#         self.lns = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in range(num_hops))
        self.dropout = dropout
        

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = x.float()
        
        x = self.preprocess(x)
        for i in range(self.num_hops):
            x = self.convs[i](x, edge_index) + x
            x = self.bns[i](x)
#             x = self.lns[i](x)
            x = self.ffnns[i](x) + x
#             if i == self.num_hops - 1 or self.num_hops - 2:
            x = F.dropout(x, p=self.dropout)

#         x = F.dropout(x, p=self.dropout)
            
            
        x = global_mean_pool(x, batch)
#         x = global_add_pool(x, batch)
        x = self.postprocess(x)
        return x
    
    
class GCN_VN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, heads=4, dropout=0.2, num_hops=2):
        super().__init__()
        self.num_hops = num_hops

        def create_ffnn(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=hidden_dim, vn=False):
            if vn:
                ffnn = nn.Sequential(
                    Linear(input_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
#                     nn.LayerNorm(2*hidden_dim),
                    nn.GELU(),
                    Linear(hidden_dim, output_dim),
                    nn.BatchNorm1d(output_dim),
#                     nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                )
            else:
                ffnn = nn.Sequential(
                    Linear(input_dim, hidden_dim),
    #                 nn.ReLU(),
                    nn.GELU(),
                    Linear(hidden_dim, output_dim)
                )                
            return ffnn

        self.vn = nn.Embedding(1, hidden_dim) ## VN embedding
        self.update_vn = nn.ModuleList([create_ffnn(hidden_dim, 2*hidden_dim, vn=True) for _ in range(num_hops)])
#             self.propogate_vn = nn.ModuleList([Linear(4*hidden_dim, hidden_dim) for _ in range(num_hops)])
        self.preprocess = create_ffnn(input_dim=input_dim)
        self.ffnns = nn.ModuleList([create_ffnn(2*hidden_dim) for _ in range(num_hops)]) # 1 for pre_process
#         self.convs = nn.ModuleList([GCNConv(hidden_dim, hidden_dim) for _ in range(num_hops)])
        self.convs = nn.ModuleList([GATv2Conv(hidden_dim, hidden_dim//heads, heads=heads) for _ in range(num_hops)])
#         GINffns = nn.ModuleList([create_ffnn() for _ in range(num_hops)])
#         self.convs = nn.ModuleList([GINConv(GINffns[i]) for i in range(num_hops)])
        self.postprocess = create_ffnn(output_dim=output_dim)
        self.bns = nn.ModuleList(nn.BatchNorm1d(2*hidden_dim) for _ in range(num_hops))
#         self.bns2 = nn.ModuleList(nn.BatchNorm1d(hidden_dim) for _ in range(num_hops))
#         self.lns = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in range(num_hops))
        self.dropout = dropout

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = x.float()
        
        x = self.preprocess(x)
        for i in range(self.num_hops):
            ## adds the mean and ffnn to send out
            vn = self.vn(torch.zeros(batch[-1].item() + 1, dtype=torch.long, device=x.device)) + global_mean_pool(x, batch) ## broadcasting here
            vn = self.update_vn[i](vn)[batch] ## reversing it to 2D
#             outvn = self.propogate_vn[i](vn) # [B, H]
            conv = self.convs[i](x, edge_index)
            pre_ffnn = self.bns[i](torch.cat([F.dropout(conv, p=self.dropout) + x, vn], dim=-1))
            x = F.dropout(self.ffnns[i](pre_ffnn), p=self.dropout) + x
#             x = self.bns2[i](x)
#             x = self.lns[i](x)
#             else:
#                 x = self.ffnns[i](x) + x
#         x = F.dropout(x, p=self.dropout)
            
        x = global_mean_pool(x, batch)
        x = self.postprocess(x)
        return x
    
    
class GIN_VN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, heads=8, dropout=0.1, num_hops=5):
        super().__init__()
        self.num_hops = num_hops
        def create_ffnn(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=hidden_dim):
            ffnn = nn.Sequential(
                Linear(input_dim, hidden_dim),
#                 nn.ReLU(),
                nn.GELU(),
                Linear(hidden_dim, output_dim)
            )
            return ffnn
        

        self.preprocess = create_ffnn(input_dim=input_dim)
        self.ffnns = nn.ModuleList([create_ffnn() for _ in range(num_hops)]) # 1 for pre_process
#         self.convs = nn.ModuleList([GATv2Conv(hidden_dim, hidden_dim//heads, heads=heads) for _ in range(num_hops)])
        GINffns = nn.ModuleList([create_ffnn() for _ in range(num_hops)])
        self.convs = nn.ModuleList([GINConv(GINffns[i]) for i in range(num_hops)])
        self.postprocess = create_ffnn(output_dim=output_dim)
        self.bns = nn.ModuleList(nn.BatchNorm1d(hidden_dim) for _ in range(num_hops))
#         self.lns = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in range(num_hops))
        self.vn = torch.nn.Parameter(torch.randn(1, 4 * hidden_dim))
        self.update_vn = nn.ModuleList([create_ffnn(output_dim=4 * hidden_dim) for _ in range(num_hops)])
        self.propogate_vn = nn.ModuleList([create_ffnn(input_dim=4 * hidden_dim) for _ in range(num_hops)])
        self.dropout = dropout
        

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = x.float()
        
        x = self.preprocess(x)
        for i in range(self.num_hops):
            vn = self.vn + self.update_vn[i](global_mean_pool(x, batch))
            outvn = self.propogate_vn[i](vn) # [B, H]
            x = self.convs[i](x, edge_index) + x
            x = self.bns[i](x)
#             x = self.lns[i](x)
#             x = self.ffnns[i](x) + x
#             if i == self.num_hops - 1 or self.num_hops - 2:
            expanded_outvn = outvn[batch] # [N, H]
            x = self.ffnns[i](expanded_outvn + x) + x
            x = F.dropout(x, p=self.dropout)

#         x = F.dropout(x, p=self.dropout)
            
            
        x = global_mean_pool(x, batch)
#         x = global_add_pool(x, batch)
        x = self.postprocess(x)
        return x
    

class GATv2_VN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, heads=8, dropout=0.1, num_hops=5):
        super().__init__()
        self.num_hops = num_hops
        def create_ffnn(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=hidden_dim):
            ffnn = nn.Sequential(
                Linear(input_dim, hidden_dim),
#                 nn.ReLU(),
                nn.GELU(),
                Linear(hidden_dim, output_dim)
            )
            return ffnn
        
        self.preprocess = create_ffnn(input_dim=input_dim)
        self.ffnns = nn.ModuleList([create_ffnn() for _ in range(num_hops)]) # 1 for pre_process
        self.convs = nn.ModuleList([GATv2Conv(hidden_dim, hidden_dim//heads, heads=heads) for _ in range(num_hops)])
        self.postprocess = create_ffnn(output_dim=output_dim)
        self.bns = nn.ModuleList(nn.BatchNorm1d(hidden_dim) for _ in range(num_hops))
        self.vn = torch.nn.Parameter(torch.randn(1, 4 * hidden_dim))
        self.update_vn = nn.ModuleList([create_ffnn(output_dim=4 * hidden_dim) for _ in range(num_hops)])
        self.propogate_vn = nn.ModuleList([create_ffnn(input_dim=4 * hidden_dim) for _ in range(num_hops)])
#         self.lns = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in range(num_hops))
        self.dropout = dropout
        

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = x.float()
        
        x = self.preprocess(x)
        for i in range(self.num_hops):
            vn = self.vn + self.update_vn[i](global_mean_pool(x, batch))
            outvn = self.propogate_vn[i](vn) # [B, H]
            x = self.convs[i](x, edge_index) + x
            x = self.bns[i](x)
#             x = self.lns[i](x)
            x = self.ffnns[i](x) + x
#             if i == self.num_hops - 1 or self.num_hops - 2:
            expanded_outvn = outvn[batch] # [N, H]
            x = self.ffnns[i](expanded_outvn + x) + x
            x = F.dropout(x, p=self.dropout)

#         x = F.dropout(x, p=self.dropout)
            
            
        x = global_mean_pool(x, batch)
#         x = global_add_pool(x, batch)
        x = self.postprocess(x)
        return x
    
class GCN_one_graph(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1, num_hops=5):
        super().__init__()
        self.num_hops = num_hops

        def create_ffnn(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=hidden_dim):
            ffnn = nn.Sequential(
                Linear(input_dim, hidden_dim),
#                 nn.ReLU(),
                nn.GELU(),
                Linear(hidden_dim, output_dim)
            )
            return ffnn
        

        self.preprocess = create_ffnn(input_dim=input_dim)
        self.ffnns = nn.ModuleList([create_ffnn() for _ in range(num_hops)]) # 1 for pre_process
        self.convs = nn.ModuleList([GCNConv(hidden_dim, hidden_dim) for _ in range(num_hops)])
        self.postprocess = create_ffnn(output_dim=output_dim)
#         self.bns = nn.ModuleList(nn.BatchNorm1d(hidden_dim) for _ in range(num_hops))
        self.lns = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in range(num_hops))
        self.dropout = dropout
        

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = x.float()
        
        
        x = self.preprocess(x)
#         x = F.dropout(x, p=self.dropout)
        for i in range(self.num_hops):
            x = self.convs[i](x, edge_index) + x
#             x = self.bns[i](x)
            x = self.lns[i](x)
            x = self.ffnns[i](x) + x
            x = F.dropout(x, p=self.dropout)
            
            
        x = self.postprocess(x)
        return x
    
class GATv2(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, heads=8, dropout=0.1, num_hops=5):
        super().__init__()
        self.num_hops = num_hops
        def create_ffnn(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=hidden_dim):
            ffnn = nn.Sequential(
                Linear(input_dim, hidden_dim),
#                 nn.ReLU(),
                nn.GELU(),
                Linear(hidden_dim, output_dim)
            )
            return ffnn
        

        self.preprocess = create_ffnn(input_dim=input_dim)
        self.ffnns = nn.ModuleList([create_ffnn() for _ in range(num_hops)]) # 1 for pre_process
        self.convs = nn.ModuleList([GATv2Conv(hidden_dim, hidden_dim//heads, heads=heads) for _ in range(num_hops)])
        self.postprocess = create_ffnn(output_dim=output_dim)
        self.bns = nn.ModuleList(nn.BatchNorm1d(hidden_dim) for _ in range(num_hops))
#         self.lns = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in range(num_hops))
        self.dropout = dropout
        

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = x.float()
        
        x = self.preprocess(x)
        for i in range(self.num_hops):
            x = self.convs[i](x, edge_index) + x
            x = self.bns[i](x)
#             x = self.lns[i](x)
            x = self.ffnns[i](x) + x
#             if i == self.num_hops - 1 or self.num_hops - 2:
            x = F.dropout(x, p=self.dropout)

#         x = F.dropout(x, p=self.dropout)
            
            
        x = global_mean_pool(x, batch)
#         x = global_add_pool(x, batch)
        x = self.postprocess(x)
        return x
    
    
class GIN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, heads=8, dropout=0.1, num_hops=5):
        super().__init__()
        self.num_hops = num_hops
        def create_ffnn(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=hidden_dim):
            ffnn = nn.Sequential(
                Linear(input_dim, hidden_dim),
#                 nn.ReLU(),
                nn.GELU(),
                Linear(hidden_dim, output_dim)
            )
            return ffnn
        

        self.preprocess = create_ffnn(input_dim=input_dim)
        self.ffnns = nn.ModuleList([create_ffnn() for _ in range(num_hops)]) # 1 for pre_process
#         self.convs = nn.ModuleList([GATv2Conv(hidden_dim, hidden_dim//heads, heads=heads) for _ in range(num_hops)])
        GINffns = nn.ModuleList([create_ffnn() for _ in range(num_hops)])
        self.convs = nn.ModuleList([GINConv(GINffns[i]) for i in range(num_hops)])
        self.postprocess = create_ffnn(output_dim=output_dim)
        self.bns = nn.ModuleList(nn.BatchNorm1d(hidden_dim) for _ in range(num_hops))
#         self.lns = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in range(num_hops))
        self.dropout = dropout
        

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = x.float()
        
        x = self.preprocess(x)
        for i in range(self.num_hops):
            x = self.convs[i](x, edge_index) + x
            x = self.bns[i](x)
#             x = self.lns[i](x)
            x = self.ffnns[i](x) + x
#             if i == self.num_hops - 1 or self.num_hops - 2:
            x = F.dropout(x, p=self.dropout)

#         x = F.dropout(x, p=self.dropout)
            
            
        x = global_mean_pool(x, batch)
#         x = global_add_pool(x, batch)
        x = self.postprocess(x)
        return x

class GATv2_one_graph(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, heads=8, dropout=0.1, num_hops=5):
        super().__init__()
        self.num_hops = num_hops
        def create_ffnn(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=hidden_dim):
            ffnn = nn.Sequential(
                Linear(input_dim, hidden_dim),
#                 nn.ReLU(),
#                 nn.GELU(),
                nn.ELU(),
                Linear(hidden_dim, output_dim),
            )
            return ffnn
        self.preprocess = create_ffnn(input_dim=input_dim)
        self.ffnns = nn.ModuleList([create_ffnn() for _ in range(num_hops)]) # 1 for pre_process
        self.convs = nn.ModuleList([GATv2Conv(hidden_dim, hidden_dim//heads, heads=heads), 
                                  GATv2Conv(hidden_dim, hidden_dim)])
        self.postprocess = create_ffnn(output_dim=output_dim)
#         self.bns = nn.ModuleList(nn.BatchNorm1d(hidden_dim) for _ in range(num_hops))
        self.lns = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in range(num_hops))
        self.dropout = dropout
        

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = x.float()
        
        
        x = self.preprocess(x)
#         x = F.dropout(x, p=self.dropout)
        for i in range(self.num_hops):
            x = self.convs[i](x, edge_index) + x
#             x = self.bns[i](x)
            x = self.lns[i](x)
            x = self.ffnns[i](x) + x
            x = F.dropout(x, p=self.dropout)
            
            
        x = self.postprocess(x)
        return x


class GIN_one_graph(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, heads=8, dropout=0.1, num_hops=5):
        super().__init__()
        self.num_hops = num_hops

        def create_ffnn(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=hidden_dim):
            ffnn = nn.Sequential(
                Linear(input_dim, hidden_dim),
#                 nn.ReLU(),
                nn.GELU(),
                Linear(hidden_dim, output_dim)
            )
            return ffnn
        

        self.preprocess = create_ffnn(input_dim=input_dim)
        self.ffnns = nn.ModuleList([create_ffnn() for _ in range(num_hops)]) # 1 for pre_process
        GINffns = nn.ModuleList([create_ffnn() for _ in range(num_hops)])
        self.convs = nn.ModuleList([GINConv(GINffns[i]) for i in range(num_hops)])
        self.postprocess = create_ffnn(output_dim=output_dim)
#         self.bns = nn.ModuleList(nn.BatchNorm1d(hidden_dim) for _ in range(num_hops))
        self.lns = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in range(num_hops))
        self.dropout = dropout
        

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = x.float()
        
        
        x = self.preprocess(x)
#         x = F.dropout(x, p=self.dropout)
        for i in range(self.num_hops):
            x = self.convs[i](x, edge_index) + x
#             x = self.bns[i](x)
            x = self.lns[i](x)
            x = self.ffnns[i](x) + x
            x = F.dropout(x, p=self.dropout)
            
            
        x = self.postprocess(x)
        return x
    
class GraphGPS(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, heads=8, dropout=0.1, num_hops=5):
        super().__init__()
        self.num_hops = num_hops
        h_dims = [hidden_dim for _ in range(num_hops)]
        def create_ffnn(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=hidden_dim):
            ffnn = nn.Sequential(
                Linear(input_dim, hidden_dim),
#                 nn.ReLU(),
                nn.GELU(),
                Linear(hidden_dim, output_dim)
            )
            return ffnn

        self.preprocess = create_ffnn(input_dim=input_dim)
#         self.ffnns = nn.ModuleList([create_ffnn() for _ in range(num_hops)]) # 1 for pre_process
#         self.convs = nn.ModuleList([GATv2Conv(h, h//heads, heads=heads) for h in h_dims])
        self.GPSConvs = nn.ModuleList([GPSConv(h, GATv2Conv(h, h//heads, heads=heads), heads=4, dropout=dropout) for h in h_dims])
#         self.convs = nn.ModuleList([GATv2Conv(h, h//heads, heads=heads) for h in h_dims])
        self.postprocess = create_ffnn(output_dim=output_dim)
#         self.bns = nn.ModuleList(nn.BatchNorm1d(h) for h in h_dims)
#         self.lns = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in range(num_hops))
        self.dropout = dropout

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = x.float()
        
        x = self.preprocess(x)
        for i in range(self.num_hops):
            x = self.GPSConvs[i](x, edge_index, batch) + x
        x = global_mean_pool(x, batch)
#         x = global_add_pool(x, batch)
        x = self.postprocess(x)
        return x