import os
from typing import Type

import torch
from tqdm import trange

from src.algos.cplex_handle import CPlexHandle
from src.algos.modules.obs_parser import GNNParser
from src.algos.network.actor_fixed_price import GNNActorFixedPrice
from src.algos.network.actor_variable_price import GNNActorVariablePrice
from src.algos.network.gnn_actor import GNNActorBase
from src.algos.network.imitate_reference import GNNActorImitateReference
from src.algos.policy.a2c_base import A2CBase
from src.algos.imitate import A2CImitating
from src.algos.policy.a2c_testing import A2CTesting
from src.algos.policy.a2c_training import A2CTraining
from src.algos.policy.a2c_training_partial import A2CTrainingPrice
from src.envs.amod_env import AMoD
from src.envs.stepper import Stepper
from src.envs.stepper_imitate import ImitateStepper
from src.misc.info import LogInfo, LogEntry
from src.misc.resource_locator import ResourceLocator
from src.misc.running_average import RunningAverage
from src.scenario.fixed_price.json_raw_data import JsonRawDataScenario


def get_actor_class(cls) -> Type[GNNActorBase]:
    return GNNActorFixedPrice if cls == 'fixed' else \
        GNNActorImitateReference if cls == 'imitate-test' else \
        GNNActorVariablePrice


def get_policy_class(cls, test) -> Type[A2CBase]:
    return A2CTesting if test else \
        A2CImitating if cls == 'imitate' or cls == 'imitate-test' else \
        A2CTrainingPrice if cls == 'price' or cls == 'fixed' else \
        A2CTraining


def get_stepper_class(cls) -> Type[Stepper]:
    return ImitateStepper if cls == 'imitate' else Stepper


class Trainer:

    def __init__(self, args, locator: ResourceLocator):
        self.locator = locator
        self.max_steps = args.max_steps
        self.pre_train = args.pre_train
        self.scenario = JsonRawDataScenario(json_file=locator.env_json_file, sd=args.seed,
                                            demand_ratio=args.demand_ratio,
                                            json_hr=args.json_hr, json_tstep=args.json_tstep,
                                            tf=self.max_steps, time_skip=args.time_skip)
        self.env = AMoD(self.scenario, beta=args.beta)
        args.cuda = not args.no_cuda and torch.cuda.is_available()
        device = torch.device("cuda" if args.cuda else "cpu")
        cls = get_actor_class(args.actor_type)
        parser = GNNParser(self.env,
                           vehicle_forecast=args.vehicle_forecast,
                           demand_forecast=args.demand_forecast)
        a2c_cls = get_policy_class(args.actor_type, args.test)
        self.model = a2c_cls(env=self.env, cls=cls, parser=parser).to(device)
        self.cplex = CPlexHandle(locator.cplex_log_folder, args.cplexpath, platform=args.platform)
        self.log = LogInfo()
        step_cls = get_stepper_class(args.actor_type)
        self.stepper = step_cls(self.cplex, self.env, self.model, self.log)
        self.max_episodes = args.max_episodes

    def train(self):
        # Initialize lists for logging
        last = 0

        running_average = RunningAverage()

        if self.pre_train != 'none' and os.path.exists(self.locator.pre_train(self.pre_train)):
            last = self.model.load_checkpoint(path=self.locator.pre_train(self.pre_train))
        epochs = trange(self.max_episodes)  # epoch iterator
        self.model.train()  # set model in train mode
        epochs.update(last)
        for episode in epochs:
            self.env.reset()  # initialize environment
            for step in range(self.max_steps):
                done, actor_reward, critic_reward = self.stepper.env_step()
                self.log.episode_data[LogEntry.gradient] = actor_reward
                self.model.save_rewards(actor_reward, critic_reward)
                if done:
                    break
            p_loss, v_loss = self.model.training_step()
            self.log.episode_data[LogEntry.policy_loss] = p_loss
            self.log.episode_data[LogEntry.value_loss] = v_loss
            self.log.finish(self.env.time)
            epochs.set_description(self.log.get_desc(episode))
            if running_average.accept(self.log.get_reward()):
                self.model.save_checkpoint(path=self.locator.save_best())
            self.log.append()
            torch.save(self.log.to_obj('train'), self.locator.train_log())

    def test(self):
        self.model.load_checkpoint(path=self.locator.test_load())
        self.model.train(False)
        epochs = trange(self.max_steps)  # epoch iterator
        # Initialize lists for logging
        reward = 0

        self.env.reset()
        for episode in epochs:
            self.stepper.env_step()
            reward += self.log.get_reward()
            self.log.finish(1)
            self.log.append()
            epochs.set_description(self.log.get_average(episode))

        print(f'Reward: {reward}')
        torch.save(self.log.to_obj('test'), self.locator.test_log())
