import torch
from torch.nn import functional as f
from torch.optim import Adam


class BaseOptim:

    def __init__(self, params, lr, gamma, eps, device):
        self.adam = Adam(params, lr=lr)
        self.gamma = gamma
        self.eps = eps
        self.device = device
        self._rewards = []

    def loss_func(self, log_prob, value, adv, r):
        raise Exception("Unimplemented")

    def append_reward(self, r):
        self._rewards.append(r)

    def step(self, saved_actions):
        cr = 0  # cumulative reward
        losses = []  # list to save actor (policy) loss
        returns = []  # list to save the true values

        # calculate the true value using rewards returned from the environment
        for r in self._rewards[::-1]:
            # calculate the discounted value
            cr = r + self.gamma * cr
            returns.insert(0, cr)

        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std(-1) + self.eps)

        for (log_prob, value), cr in zip(saved_actions, returns):
            advantage = cr - value.item()

            # calculate actor (policy) loss
            losses.append(self.loss_func(log_prob, value, advantage, cr))

        # take gradient steps
        self.adam.zero_grad()
        a_loss = torch.stack(losses).sum()
        a_loss.backward()
        self.adam.step()
        del self._rewards[:]
        return a_loss.item()


class ActorOptim(BaseOptim):

    def __init__(self, params, lr, gamma, eps, device):
        super().__init__(params, lr, gamma, eps, device)

    def loss_func(self, log_prob, value, adv, r):
        # calculate actor (policy) loss
        return -log_prob * adv


class CriticOptim(BaseOptim):

    def __init__(self, params, lr, gamma, eps, device):
        super().__init__(params, lr, gamma, eps, device)

    def loss_func(self, log_prob, value, adv, r):
        # calculate critic (value) loss using L1 smooth loss
        return f.smooth_l1_loss(value, torch.tensor([r]).to(self.device))
