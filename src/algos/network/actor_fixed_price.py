import torch
from torch import nn

from src.algos.network.gnn_actor import GNNActorBase


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
