import torch
from torch import nn
from torch.nn import functional as f
from torch_geometric.nn import GCNConv

from src.algos.modules.true_bilinear import TrueBilinear
from src.algos.network.gnn_actor import GNNActorBase


class GNNActorVariablePrice(GNNActorBase):
    """
    Actor pi(a_t | s_t) parametrizing the concentration parameters of a Dirichlet Policy.
    """

    def __init__(self, in_channels, mid_channels, nregion, variance, pmid=8):
        super().__init__(in_channels, mid_channels, nregion, variance)

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
