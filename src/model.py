# src/model.py
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from src.config import Config

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=Config.HIDDEN_DIM, out_channels=Config.NUM_CLASSES):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        self.dropout = torch.nn.Dropout(p=Config.DROPOUT)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)