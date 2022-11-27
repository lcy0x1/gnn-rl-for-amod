import torch
from torch_geometric.data import Data
from torch_geometric.utils import grid

from src.envs.amod_env import AMoD


class GNNParser:
    """
    Parser converting raw environment observations to agent inputs (s_t).
    """

    def __init__(self, env: AMoD, vehicle_forecast=10, demand_forecast=10, grid_h=4, grid_w=4):
        super().__init__()
        self.env = env
        self.vehicle_forecast = vehicle_forecast
        self.demand_forecast = demand_forecast
        self.s = 1 / env.data.total_acc
        self.grid_h = grid_h
        self.grid_w = grid_w

    def parse_obs(self):
        size = self.env.nregion
        time = self.env.time

        x = torch.cat(
            (
                torch.tensor([self.env.data.acc[n][time + 1] * self.s for n in range(size)]).view(1, 1, size).float(),
                torch.tensor([[(self.env.data.acc[n][time + 1] + self.env.data.dacc[n][t]) * self.s
                               for n in range(size)] for t in range(time + 1, time + self.vehicle_forecast + 1)]
                             ).view(1, self.vehicle_forecast, size).float(),
                torch.tensor([[sum([self.env.data.get_principal_demand(i, j, t) * self.s
                                    for j in range(size)]) for i in range(size)]
                              for t in range(time + 1, time + self.demand_forecast + 1)]
                             ).view(1, self.demand_forecast, size).float(),
                torch.tensor([[sum(self.env.data.get_demand(o, d, time) * self.s
                                   for d in range(size))] for o in range(size)]
                             ).view(1, 1, size).float()
            ), dim=1).squeeze(0).view(self.width(), size).T
        edge_index, pos_coord = grid(height=self.grid_h, width=self.grid_w)
        data = Data(x, edge_index)
        return data

    def width(self):
        return 1 + self.vehicle_forecast + self.demand_forecast + 1
