import torch
from torch import nn
from torch.nn import functional as f
from torch_geometric.nn import GCNConv


class GNNActorBase(nn.Module):
    """
    Actor pi(a_t | s_t) parametrizing the concentration parameters of a Dirichlet Policy.
    """

    def __init__(self, in_channels, mid_channels, nregion, gamma_rate):
        super().__init__()

        self.conv1 = GCNConv(in_channels, in_channels)
        self.lin1 = nn.Linear(in_channels, mid_channels)
        self.lin2 = nn.Linear(mid_channels, mid_channels)
        self.lin3 = nn.Linear(mid_channels, 1)

        self.gamma_rate = nn.Parameter(torch.ones((nregion, nregion)) * gamma_rate, requires_grad=True)

    def forward_price(self, data):
        out = f.relu(self.conv1(data.x, data.edge_index))
        x = out + data.x
        x = f.relu(self.lin1(x))
        x = f.relu(self.lin2(x))
        x = self.lin3(x)
        return x
