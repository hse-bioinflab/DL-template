import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GraphConv, GATConv, GATv2Conv, SAGEConv

class GraphZ(torch.nn.Module):
    def __init__(self):
        super(GraphZ, self).__init__()
        self.conv1 = GraphConv(1058, 500)
        self.conv2 = GraphConv(500, 250)
        self.conv3 = GraphConv(250, 100)
        self.conv4 = GraphConv(100, 2)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x)
        x = self.conv4(x, edge_index)

        return F.log_softmax(x, dim=1)
