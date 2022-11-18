import os

import torch
from tqdm import trange

from src.algos.a2c_gnn import A2C
from src.algos.gnn_imitate import GNNActorImitateReference
from src.algos.gnn_networks import GNNActorVariablePrice, GNNActorFixedPrice
from src.algos.obs_parser import GNNParser
from src.algos.cplex_handle import CPlexHandle
from src.envs.amod_env import AMoD
from src.envs.running_average import RunningAverage
from src.envs.stepper import Stepper, ImitateStepper
from src.misc.info import LogInfo, LogEntry
from src.misc.resource_locator import ResourceLocator
from src.scenario.fixed_price.json_raw_data import JsonRawDataScenario


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
        cls = GNNActorFixedPrice if args.actor_type == 'fixed' else GNNActorVariablePrice
        parser = GNNParser(self.env,
                           vehicle_forecast=args.vehicle_forecast,
                           demand_forecast=args.demand_forecast)
        self.model = A2C(env=self.env, cls=cls, parser=parser).to(device)
        self.cplex = CPlexHandle(locator.cplex_log_folder, args.cplexpath, platform=args.platform)
        self.log = LogInfo()
        step_cls = ImitateStepper if args.actor_type == 'imitate' else Stepper
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
                done, backprop = self.stepper.env_step()
                self.log.episode_data[LogEntry.gradient] += backprop
                self.model.rewards.append(backprop)
                if done:
                    break
            self.model.training_step()
            self.log.finish(self.env.time)
            epochs.set_description(self.log.get_desc(episode))
            if running_average.accept(self.log.get_reward()):
                self.model.save_checkpoint(path=self.locator.save_best())
            self.log.append()
            self.model.log(self.log.to_obj('train'), path=self.locator.train_log())

    def test(self):
        self.model.load_checkpoint(path=self.locator.test_load())
        self.model.train(False)
        epochs = trange(self.max_steps)  # epoch iterator
        # Initialize lists for logging
        reward = 0

        self.env.reset()
        for episode in epochs:
            done = self.stepper.env_step()
            reward += self.log.get_reward()
            self.log.finish(1)
            self.log.append()
            epochs.set_description(self.log.get_average(episode))
            # Log KPIs
            if done:
                break

        print(f'Reward: {reward}')
        self.model.log(self.log.to_obj('test'), path=self.locator.test_log())
