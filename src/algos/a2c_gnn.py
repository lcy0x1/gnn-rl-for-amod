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
from typing import Type

import numpy as np

from torch.distributions import Dirichlet, Gamma
from torch.nn import functional as F
from collections import namedtuple

from src.algos.gnn_networks import *
from src.algos.obs_parser import GNNParser
from src.envs.amod_env import AMoD

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])
args = namedtuple('args', ('render', 'gamma', 'log_interval'))
args.render = True
args.gamma = 0.97
args.log_interval = 10


class A2C(nn.Module):
    """
    Advantage Actor Critic algorithm for the AMoD control problem. 
    """

    def __init__(self, env: AMoD, parser: GNNParser, hidden_size=32,
                 eps=np.finfo(np.float32).eps.item(),
                 device=torch.device("cpu"),
                 cls: Type[GNNActorBase] = GNNActorVariablePrice, gamma_rate: float = 20):
        super(A2C, self).__init__()
        self.env = env
        self.eps = eps
        self.input_size = parser.width()
        self.hidden_size = hidden_size
        self.device = device

        self.actor = cls(self.input_size, self.hidden_size, env.nregion, gamma_rate)
        self.critic = GNNCritic(self.input_size, self.hidden_size)
        self.obs_parser = parser

        self.optimizers = self.configure_optimizers()

        # action & reward buffer
        self.saved_actions = []
        self.rewards = []
        self.to(self.device)

    def forward(self, obs, jitter=1e-20):
        """
        forward of both actor and critic
        """
        # parse raw environment data in model format
        x = self.obs_parser.parse_obs(obs).to(self.device)

        # actor: computes concentration parameters of a Dirichlet distribution
        a_out, raw_price = self.actor.forward(x)
        concentration = F.softplus(a_out).reshape(-1) + jitter
        price = F.softplus(raw_price + 1) + jitter

        # critic: estimates V(s_t)
        value = self.critic(x)
        return concentration, price, value

    def select_action(self, obs):
        vehicle_vec, price_mat, value = self.forward(obs)

        vehicle_dist = Dirichlet(vehicle_vec)

        if not self.training:
            vehicle_action = vehicle_dist.mean
            return list(vehicle_action.detach().cpu().numpy()), price_mat.detach().cpu().numpy()

        vehicle_action = vehicle_dist.sample()
        price_dist = Gamma(price_mat * self.actor.gamma_rate, self.actor.gamma_rate)
        price_action = price_dist.sample()

        self.saved_actions.append(SavedAction(
            vehicle_dist.log_prob(vehicle_action) +
            price_dist.log_prob(price_action),
            value))

        return list(vehicle_action.cpu().numpy()), price_action.cpu().numpy()

    def training_step(self):
        R = 0
        saved_actions = self.saved_actions
        policy_losses = []  # list to save actor (policy) loss
        value_losses = []  # list to save critic (value) loss
        returns = []  # list to save the true values

        # calculate the true value using rewards returned from the environment
        for r in self.rewards[::-1]:
            # calculate the discounted value
            R = r + args.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + self.eps)

        for (log_prob, value), R in zip(saved_actions, returns):
            advantage = R - value.item()

            # calculate actor (policy) loss 
            policy_losses.append(-log_prob * advantage)

            # calculate critic (value) loss using L1 smooth loss
            value_losses.append(F.smooth_l1_loss(value, torch.tensor([R]).to(self.device)))

        # take gradient steps
        self.optimizers['a_optimizer'].zero_grad()
        a_loss = torch.stack(policy_losses).sum()
        a_loss.backward()
        self.optimizers['a_optimizer'].step()

        self.optimizers['c_optimizer'].zero_grad()
        v_loss = torch.stack(value_losses).sum()
        v_loss.backward()
        self.optimizers['c_optimizer'].step()

        # reset rewards and action buffer
        del self.rewards[:]
        del self.saved_actions[:]

    def configure_optimizers(self):
        optimizers = dict()
        actor_params = list(self.actor.parameters())
        critic_params = list(self.critic.parameters())
        optimizers['a_optimizer'] = torch.optim.Adam(actor_params, lr=1e-3)
        optimizers['c_optimizer'] = torch.optim.Adam(critic_params, lr=1e-3)
        return optimizers

    def save_checkpoint(self, path='ckpt.pth'):
        checkpoint = dict()
        checkpoint['model'] = self.state_dict()
        for key, value in self.optimizers.items():
            checkpoint[key] = value.state_dict()
        torch.save(checkpoint, path)

    def load_checkpoint(self, path='ckpt.pth'):
        checkpoint = torch.load(path)
        incp = self.load_state_dict(checkpoint['model'], strict=False)
        if not incp.missing_keys and not incp.unexpected_keys:
            for key, value in self.optimizers.items():
                self.optimizers[key].load_state_dict(checkpoint[key])
        return checkpoint['episode'] if 'episode' in checkpoint else 0

    def log(self, log_dict, path='log.pth'):
        torch.save(log_dict, path)
