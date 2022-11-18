import torch

from src.algos.gnn_networks import GNNActorBase


class GNNActorImitateReference(GNNActorBase):

    def __init__(self, in_channels, mid_channels, nregion, gamma_rate):
        super().__init__(in_channels, mid_channels, nregion, gamma_rate)
        self.nregion = nregion

    def forward(self, data):
        x1 = self.forward_price(data)
        demand = data.x[:, 11]
        acc = data.x[:, 1]
        prices = -0.2 - torch.log(acc / demand)
        n = self.nregion
        prices = prices.view((n, 1)).matmul(torch.ones(n).view((1, n)))
        return x1, prices
