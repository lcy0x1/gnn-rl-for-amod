from typing import Type

import numpy as np
import torch

from src.algos.modules.obs_parser import GNNParser
from src.algos.network.actor_variable_price import GNNActorVariablePrice
from src.algos.network.gnn_actor import GNNActorBase
from src.algos.optim.actor_optim import ActorOptim
from src.algos.optim.critic_optim import CriticOptim
from src.algos.policy.a2c_base import A2CBase, SavedAction
from src.envs.amod_env import AMoD


class A2CTrainingPrice(A2CBase):

    def __init__(self, env: AMoD, parser: GNNParser, hidden_size=32, eps=np.finfo(np.float32).eps.item(),
                 device=torch.device("cpu"), cls: Type[GNNActorBase] = GNNActorVariablePrice, variance: float = 1):
        super().__init__(env, parser, hidden_size, eps, device, cls, variance)

    def select_action(self, obs):
        vehicle_vec, price_mat, value = self.forward(obs)
        vehicle = self.actor.vehicle_dist(vehicle_vec, train=False)
        price, plp = self.actor.price_dist(price_mat, train=True)
        self.saved_actions.append(SavedAction(plp, value))
        return list(vehicle.cpu().numpy()), price.cpu().numpy()

    def configure_optimizers(self):
        actor_params = list(self.actor.parameters())
        critic_params = list(self.critic.parameters())
        a = ActorOptim(actor_params, lr=1e-4, gamma=0.5, eps=self.eps, device=self.device)
        c = CriticOptim(critic_params, lr=1e-3, gamma=0.97, eps=self.eps, device=self.device)
        return a, c


class A2CTrainingRebalance(A2CBase):

    def __init__(self, env: AMoD, parser: GNNParser, hidden_size=32, eps=np.finfo(np.float32).eps.item(),
                 device=torch.device("cpu"), cls: Type[GNNActorBase] = GNNActorVariablePrice, variance: float = 2000):
        super().__init__(env, parser, hidden_size, eps, device, cls, variance)

    def select_action(self, obs):
        vehicle_vec, price_mat, value = self.forward(obs)
        vehicle, vlp = self.actor.vehicle_dist(vehicle_vec, train=True)
        price = self.actor.price_dist(price_mat, train=False)
        self.saved_actions.append(SavedAction(vlp, value))
        return list(vehicle.cpu().numpy()), price.cpu().numpy()
