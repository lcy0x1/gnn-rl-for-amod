import numpy as np

from src.algos.cplex_handle import CPlexHandle
from src.algos.imitate import A2CImitating
from src.algos.network.imitate_reference import GNNActorImitateReference
from src.algos.policy.a2c_testing import A2CTesting
from src.envs.amod_env import AMoD
from src.envs.stepper import Stepper
from src.misc.info import LogInfo


class ImitateStepper(Stepper):

    def __init__(self, cplex: CPlexHandle, env: AMoD, model: A2CImitating, log: LogInfo):
        super().__init__(cplex, env, model, log)
        self.reference = A2CTesting(env=env, cls=GNNActorImitateReference, parser=model.obs_parser)
        self.reference.train(mode=False)
        self.diff = 0

    def select_action(self, obs):
        _, ref_price = self.reference.select_action(obs)
        acc, price = self.model.select_action(ref_price)
        self.diff = -np.square(ref_price - price).sum()
        return acc, price

    def env_step(self):
        done, _, reward = super().env_step()
        return done, self.diff, reward
