from typing import Type

import numpy as np
import torch
from torch.distributions import Dirichlet

from src.algos.modules.obs_parser import GNNParser
from src.algos.network.actor_variable_price import GNNActorVariablePrice
from src.algos.network.gnn_actor import GNNActorBase
from src.algos.policy.a2c_base import A2CBase
from src.envs.amod_env import AMoD


class A2CTesting(A2CBase):

    def __init__(self, env: AMoD, parser: GNNParser, hidden_size=32, eps=np.finfo(np.float32).eps.item(),
                 device=torch.device("cpu"), cls: Type[GNNActorBase] = GNNActorVariablePrice, gamma_rate: float = 2000):
        super().__init__(env, parser, hidden_size, eps, device, cls, gamma_rate)

    def select_action(self, obs):
        vehicle_vec, price_mat, value = self.forward(obs)
        vehicle_dist = Dirichlet(vehicle_vec)
        vehicle_action = vehicle_dist.mean
        return list(vehicle_action.detach().cpu().numpy()), price_mat.detach().cpu().numpy()
