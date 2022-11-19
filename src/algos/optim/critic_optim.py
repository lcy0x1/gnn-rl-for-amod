import torch
from torch.nn import functional as f

from src.algos.optim.base_optim import BaseOptim


class CriticOptim(BaseOptim):

    def __init__(self, params, lr, gamma, eps, device):
        super().__init__(params, lr, gamma, eps, device)

    def loss_func(self, log_prob, value, adv, r):
        # calculate critic (value) loss using L1 smooth loss
        return f.smooth_l1_loss(value, torch.tensor([r]).to(self.device))
