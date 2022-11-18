import torch
from torch import nn
from torch.nn import functional as f
from torch_geometric.nn import GCNConv


class GNNCritic(nn.Module):
    """
    Critic parametrizing the value function estimator V(s_t).
    """

    def __init__(self, in_channels, mid_channels):
        super().__init__()

        self.conv1 = GCNConv(in_channels, in_channels)
        self.lin1 = nn.Linear(in_channels, mid_channels)
        self.lin2 = nn.Linear(mid_channels, mid_channels)
        self.lin3 = nn.Linear(mid_channels, 1)

    def forward(self, data):
        out = f.relu(self.conv1(data.x, data.edge_index))
        x = out + data.x
        x = torch.sum(x, dim=0)
        x = f.relu(self.lin1(x))
        x = f.relu(self.lin2(x))
        x = self.lin3(x)
        return x