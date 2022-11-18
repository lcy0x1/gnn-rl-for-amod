import torch
from torch import nn
from torch.nn import functional as f
from torch_geometric.nn import GCNConv

from src.algos.true_bilinear import TrueBilinear


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


class GNNActorVariablePrice(GNNActorBase):
    """
    Actor pi(a_t | s_t) parametrizing the concentration parameters of a Dirichlet Policy.
    """

    def __init__(self, in_channels, mid_channels, nregion, gamma_rate, pmid=8):
        super().__init__(in_channels, mid_channels, nregion, gamma_rate)

        self.convp = GCNConv(in_channels, in_channels)
        self.lin1p = nn.Linear(in_channels, pmid)
        self.bilinp = TrueBilinear(pmid, pmid, nregion, nregion, bias=False)
        self.prices = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, data):
        acc = self.forward_price(data)
        out = f.relu(self.convp(data.x, data.edge_index))
        x = out + data.x
        x = f.relu(self.lin1p(x))
        x = self.bilinp.forward(x, x) + self.prices
        return acc, x


class GNNActorFixedPrice(GNNActorBase):
    """
    Actor pi(a_t | s_t) parametrizing the concentration parameters of a Dirichlet Policy.
    """

    def __init__(self, in_channels, mid_channels, nregion, gamma_rate):
        super().__init__(in_channels, mid_channels, nregion, gamma_rate)

        self.prices = nn.Parameter(torch.zeros((nregion, nregion)), requires_grad=True)

    def forward(self, data):
        x = self.forward_price(data)
        return x, self.prices


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
