from typing import Type

import numpy as np
import torch
from torch.distributions import Dirichlet, Gamma

from src.algos.modules.obs_parser import GNNParser
from src.algos.network.actor_variable_price import GNNActorVariablePrice
from src.algos.network.gnn_actor import GNNActorBase
from src.algos.optim.base_optim import BaseOptim
from src.algos.optim.critic_optim import CriticOptim
from src.algos.policy.a2c_base import A2CBase, SavedAction
from src.envs.amod_env import AMoD


class ImitateOptim(BaseOptim):

    def __init__(self, params, lr, gamma, eps, device):
        super().__init__(params, lr, gamma, eps, device)

    def loss_func(self, dist, value, adv, r):
        # calculate actor (policy) loss
        return dist


class A2CImitating(A2CBase):

    def __init__(self, env: AMoD, parser: GNNParser, hidden_size=32, eps=np.finfo(np.float32).eps.item(),
                 device=torch.device("cpu"), cls: Type[GNNActorBase] = GNNActorVariablePrice, gamma_rate: float = 2000):
        super().__init__(env, parser, hidden_size, eps, device, cls, gamma_rate)

    def select_action(self, ref):
        vehicle_vec, price_mat, value = self.forward(ref)
        vehicle_dist = Dirichlet(vehicle_vec)
        vehicle_action = vehicle_dist.mean
        price_dist = Gamma(price_mat * self.actor.gamma_rate, self.actor.gamma_rate)
        price_action = price_dist.mean
        dist = torch.square(torch.tensor(ref) - price_action).sum()
        self.saved_actions.append(SavedAction(dist, value))
        return list(vehicle_action.detach().cpu().numpy()), price_action.detach().cpu().numpy()

    def configure_optimizers(self):
        actor_params = list(self.actor.parameters())
        critic_params = list(self.critic.parameters())
        a = ImitateOptim(actor_params, lr=3e-4, gamma=0, eps=self.eps, device=self.device)
        c = CriticOptim(critic_params, lr=1e-3, gamma=0.97, eps=self.eps, device=self.device)
        return a, c
