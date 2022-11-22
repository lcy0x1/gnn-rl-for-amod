import torch
from torch import nn
from torch.distributions import Normal, Dirichlet
from torch.nn import functional as f
from torch_geometric.nn import GCNConv


class GNNActorBase(nn.Module):
    """
    Actor pi(a_t | s_t) parametrizing the concentration parameters of a Dirichlet Policy.
    """

    def __init__(self, in_channels, mid_channels, nregion, variance):
        super().__init__()

        self.conv1 = GCNConv(in_channels, in_channels)
        self.lin1 = nn.Linear(in_channels, mid_channels)
        self.lin2 = nn.Linear(mid_channels, mid_channels)
        self.lin3 = nn.Linear(mid_channels, 1)

        self.variance = nn.Parameter(torch.ones((nregion, nregion)) * variance, requires_grad=True)

    def forward_price(self, data):
        out = f.relu(self.conv1(data.x, data.edge_index))
        x = out + data.x
        x = f.relu(self.lin1(x))
        x = f.relu(self.lin2(x))
        x = self.lin3(x)
        return x

    def price_dist(self, price, train=True):
        if train:
            dist = Normal(price, self.variance)
            sample = dist.sample()
            log_prob = dist.log_prob(sample).sum(-1)
            return torch.tanh(sample) + 1, log_prob
        else:
            return torch.tanh(price.detach()) + 1

    def vehicle_dist(self, vehicle, train=True):
        dist = Dirichlet(vehicle)
        if train:
            sample = dist.sample()
            return sample, dist.log_prob(sample)
        else:
            return dist.mean.detach()
