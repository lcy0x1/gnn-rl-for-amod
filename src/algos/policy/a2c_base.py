"""
A2C-GNN
-------
This file contains the A2C-GNN specifications. In particular, we implement:
(1) GNNParser
    Converts raw environment observations to agent inputs (s_t).
(2) GNNActor:
    Policy parametrized by Graph Convolution Networks (Section III-C in the paper)
(3) GNNCritic:
    Critic parametrized by Graph Convolution Networks (Section III-C in the paper)
(4) A2C:
    Advantage Actor Critic algorithm using a GNN parametrization for both Actor and Critic.
"""
from collections import namedtuple
from typing import Type

import numpy as np

from src.algos.modules.obs_parser import GNNParser
from src.algos.network.actor_variable_price import GNNActorVariablePrice
from src.algos.network.gnn_actor import *
from src.algos.network.gnn_critic import GNNCritic
from src.algos.optim.actor_optim import ActorOptim
from src.algos.optim.critic_optim import CriticOptim
from src.envs.amod_env import AMoD

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])


class A2CBase(nn.Module):
    """
    Advantage Actor Critic algorithm for the AMoD control problem. 
    """

    def __init__(self, env: AMoD, parser: GNNParser, hidden_size=32, eps=np.finfo(np.float32).eps.item(),
                 device=torch.device("cpu"), cls: Type[GNNActorBase] = GNNActorVariablePrice, gamma_rate: float = 2000):
        super(A2CBase, self).__init__()
        self.env = env
        self.eps = eps
        self.input_size = parser.width()
        self.hidden_size = hidden_size
        self.device = device

        self.actor = cls(self.input_size, self.hidden_size, env.nregion, gamma_rate)
        self.critic = GNNCritic(self.input_size, self.hidden_size)
        self.obs_parser = parser

        self.actor_optim, self.critic_optim = self.configure_optimizers()

        # action & reward buffer
        self.saved_actions = []
        self.to(self.device)

    def forward(self, _, jitter=1e-20):
        """
        forward of both actor and critic
        """
        # parse raw environment data in model format
        x = self.obs_parser.parse_obs().to(self.device)

        # actor: computes concentration parameters of a Dirichlet distribution
        a_out, raw_price = self.actor.forward(x)
        concentration = f.softplus(a_out).reshape(-1) + jitter
        price = f.softplus(raw_price + 1) + jitter

        # critic: estimates V(s_t)
        value = self.critic(x)
        return concentration, price, value

    def select_action(self, obs):
        raise Exception("select_action not implemented yet")

    def save_rewards(self, ar, cr):
        self.actor_optim.append_reward(ar)
        self.critic_optim.append_reward(cr)

    def training_step(self):
        pl = self.actor_optim.step(self.saved_actions)
        vl = self.critic_optim.step(self.saved_actions)
        # reset rewards and action buffer
        del self.saved_actions[:]
        return pl, vl

    def configure_optimizers(self):
        actor_params = list(self.actor.parameters())
        critic_params = list(self.critic.parameters())
        a = ActorOptim(actor_params, lr=1e-3, gamma=0.97, eps=self.eps, device=self.device)
        c = CriticOptim(critic_params, lr=1e-3, gamma=0.97, eps=self.eps, device=self.device)
        return a, c

    def save_checkpoint(self, path='ckpt.pth'):
        checkpoint = dict()
        checkpoint['model'] = self.state_dict()
        checkpoint['a_optimizer'] = self.actor_optim.adam.state_dict()
        checkpoint['c_optimizer'] = self.critic_optim.adam.state_dict()
        torch.save(checkpoint, path)

    def load_checkpoint(self, path='ckpt.pth'):
        checkpoint = torch.load(path)
        if 'actor.gamma_rate' in checkpoint['model']:
            del checkpoint['model']['actor.gamma_rate']
        incp = self.load_state_dict(checkpoint['model'], strict=False)
        if not incp.missing_keys and not incp.unexpected_keys:
            self.actor_optim.adam.load_state_dict(checkpoint['a_optimizer'])
            self.critic_optim.adam.load_state_dict(checkpoint['c_optimizer'])
        return checkpoint['episode'] if 'episode' in checkpoint else 0
