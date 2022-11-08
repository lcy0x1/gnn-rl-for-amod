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

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from torch.distributions import Dirichlet
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric.utils import grid
from collections import namedtuple

from src.envs.amod_env import AMoD

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])
args = namedtuple('args', ('render', 'gamma', 'log_interval'))
args.render = True
args.gamma = 0.97
args.log_interval = 10


#########################################
############## PARSER ###################
#########################################

class GNNParser:
    """
    Parser converting raw environment observations to agent inputs (s_t).
    """

    def __init__(self, env: AMoD, vehicle_forecast=10, demand_forecast=10, grid_h=4, grid_w=4, scale_factor=0.01):
        super().__init__()
        self.env = env
        self.vehicle_forecast = vehicle_forecast
        self.demand_forecast = demand_forecast
        self.s = scale_factor
        self.grid_h = grid_h
        self.grid_w = grid_w

    def parse_obs(self, obs):
        size = self.env.nregion
        time = self.env.time

        x = torch.cat(
            (
                torch.tensor([self.env.data.acc[n][time + 1] * self.s for n in range(size)]).view(1, 1, size).float(),
                torch.tensor([[(self.env.data.acc[n][time + 1] + self.env.data.dacc[n][t]) * self.s
                               for n in range(size)] for t in range(time + 1, time + self.vehicle_forecast + 1)])
                    .view(1, self.vehicle_forecast, size).float(),
                torch.tensor([[sum([(self.env.get_demand_input(i, j, t)) *
                                    (self.env.data.price[i, j][t]) * self.s
                                    for j in range(size)]) for i in range(size)]
                              for t in range(time + 1, time + self.demand_forecast + 1)])
                    .view(1, self.demand_forecast, size).float()
            ), dim=1).squeeze(0).view(1 + self.vehicle_forecast + self.demand_forecast, size).T
        edge_index, pos_coord = grid(height=self.grid_h, width=self.grid_w)
        data = Data(x, edge_index)
        return data

    def width(self):
        return 1 + self.vehicle_forecast + self.demand_forecast


#########################################
############## ACTOR ####################
#########################################
class GNNActor(nn.Module):
    """
    Actor pi(a_t | s_t) parametrizing the concentration parameters of a Dirichlet Policy.
    """

    def __init__(self, in_channels, mid_channels):
        super().__init__()

        self.conv1 = GCNConv(in_channels, in_channels)
        self.lin1 = nn.Linear(in_channels, mid_channels)
        self.lin2 = nn.Linear(mid_channels, mid_channels)
        self.lin3 = nn.Linear(mid_channels, 1)

    def forward(self, data):
        out = F.relu(self.conv1(data.x, data.edge_index))
        x = out + data.x
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = self.lin3(x)
        return x


#########################################
############## CRITIC ###################
#########################################

class GNNCritic(nn.Module):
    """
    Critic parametrizing the value function estimator V(s_t).
    """

    def __init__(self, in_channels, mid_channels):
        super().__init__()

        self.conv1 = GCNConv(in_channels, in_channels)
        self.lin1 = nn.Linear(in_channels, mid_channels)
        self.lin2 = nn.Linear(mid_channels, mid_channels)
        self.lin3 = nn.Linear(mid_channels, 1)

    def forward(self, data):
        out = F.relu(self.conv1(data.x, data.edge_index))
        x = out + data.x
        x = torch.sum(x, dim=0)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = self.lin3(x)
        return x


#########################################
############## A2C AGENT ################
#########################################

class A2C(nn.Module):
    """
    Advantage Actor Critic algorithm for the AMoD control problem. 
    """

    def __init__(self, env, parser: GNNParser, hidden_size=32,
                 eps=np.finfo(np.float32).eps.item(),
                 device=torch.device("cpu")):
        super(A2C, self).__init__()
        self.env = env
        self.eps = eps
        self.input_size = parser.width()
        self.hidden_size = hidden_size
        self.device = device

        self.actor = GNNActor(self.input_size, self.hidden_size)
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
        a_out = self.actor(x)
        concentration = F.softplus(a_out).reshape(-1) + jitter

        # critic: estimates V(s_t)
        value = self.critic(x)
        return concentration, value

    def select_action(self, obs):
        concentration, value = self.forward(obs)

        m = Dirichlet(concentration)

        action = m.sample()
        self.saved_actions.append(SavedAction(m.log_prob(action), value))
        return list(action.cpu().numpy())

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
        self.load_state_dict(checkpoint['model'])
        for key, value in self.optimizers.items():
            self.optimizers[key].load_state_dict(checkpoint[key])

    def log(self, log_dict, path='log.pth'):
        torch.save(log_dict, path)
