import torch
from torch.nn import functional as F

from src.algos.network.gnn_actor import GNNActorBase


class GNNActorImitateReference(GNNActorBase):

    def __init__(self, in_channels, mid_channels, nregion, gamma_rate):
        super().__init__(in_channels, mid_channels, nregion, gamma_rate)
        self.nregion = nregion

    def forward(self, data):
        demand = data.x[:, 11]
        acc = data.x[:, 1]
        prices = 0.5 - torch.log(F.softplus(acc) / F.softplus(demand))
        n = self.nregion
        prices = prices.view((n, 1)).matmul(torch.ones(n).view((1, n)))
        return self.forward_price(data), prices
