from src.algos.optim.base_optim import BaseOptim


class ActorOptim(BaseOptim):

    def __init__(self, params, lr, gamma, eps, device):
        super().__init__(params, lr, gamma, eps, device)

    def loss_func(self, log_prob, value, adv, r):
        # calculate actor (policy) loss
        return -log_prob * adv
