from typing import Type

import numpy as np
import torch

from src.algos.modules.obs_parser import GNNParser
from src.algos.network.actor_variable_price import GNNActorVariablePrice
from src.algos.network.gnn_actor import GNNActorBase
from src.algos.policy.a2c_base import A2CBase
from src.envs.amod_env import AMoD


class A2CTesting(A2CBase):

    def __init__(self, env: AMoD, parser: GNNParser, hidden_size=32, eps=np.finfo(np.float32).eps.item(),
                 device=torch.device("cpu"), cls: Type[GNNActorBase] = GNNActorVariablePrice, variance: float = 1):
        super().__init__(env, parser, hidden_size, eps, device, cls, variance)

    def select_action(self, obs):
        vehicle_vec, price_mat, value = self.forward(obs)
        vehicle = self.actor.vehicle_dist(vehicle_vec, train=False)
        price = self.actor.price_dist(price_mat, train=False)
        return list(vehicle.cpu().numpy()), price.cpu().numpy()
