import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GCNConv


class GNNActorBase(nn.Module):
    """
    Actor pi(a_t | s_t) parametrizing the concentration parameters of a Dirichlet Policy.
    """

    def __init__(self, in_channels, mid_channels):
        super().__init__()

        self.conv1 = GCNConv(in_channels, in_channels)
        self.lin1 = nn.Linear(in_channels, mid_channels)
        self.lin2 = nn.Linear(mid_channels, mid_channels)
        self.lin3 = nn.Linear(mid_channels, 1)

    def forward_price(self, data):
        out = F.relu(self.conv1(data.x, data.edge_index))
        x0 = out + data.x
        x0 = F.relu(self.lin1(x0))
        x1 = F.relu(self.lin2(x0))
        x1 = self.lin3(x1)
        return x0, x1


class GNNActorVariablePrice(GNNActorBase):
    """
    Actor pi(a_t | s_t) parametrizing the concentration parameters of a Dirichlet Policy.
    """

    def __init__(self, in_channels, mid_channels, nregion):
        super().__init__(in_channels, mid_channels)
        self.lin4 = nn.Linear(mid_channels, mid_channels)
        self.bilin = nn.Bilinear(mid_channels, mid_channels, nregion)

    def forward(self, data):
        x0, x1 = self.forward_price(data)
        x2 = F.relu(self.lin4(x0))
        x2 = self.bilin(x2, x2)
        return x1, x2


class GNNActorFixedPrice(GNNActorBase):
    """
    Actor pi(a_t | s_t) parametrizing the concentration parameters of a Dirichlet Policy.
    """

    def __init__(self, in_channels, mid_channels, nregion):
        super().__init__(in_channels, mid_channels)

        self.prices = nn.Parameter(torch.zeros((nregion, nregion)), requires_grad=True)

    def forward(self, data):
        _, x1 = self.forward_price(data)
        return x1, self.prices


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
        out = F.relu(self.conv1(data.x, data.edge_index))
        x = out + data.x
        x = torch.sum(x, dim=0)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = self.lin3(x)
        return x