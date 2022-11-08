import numpy as np
import torch
from tqdm import trange

from src.algos.a2c_gnn import A2C
from src.algos.cplex_handle import CPlexHandle
from src.envs.amod_env import AMoD
from src.misc.info import LogInfo
from src.misc.resource_locator import ResourceLocator
from src.misc.utils import dictsum
from src.scenario.fixed_price.json_raw_data import JsonRawDataScenario


class Trainer:

    def __init__(self, args, locator: ResourceLocator):
        self.locator = locator
        self.scenario = JsonRawDataScenario(json_file=locator.env_json_file, sd=args.seed,
                                            demand_ratio=args.demand_ratio,
                                            json_hr=args.json_hr, json_tstep=args.json_tsetp)
        self.env = AMoD(self.scenario, beta=args.beta)
        args.cuda = not args.no_cuda and torch.cuda.is_available()
        device = torch.device("cuda" if args.cuda else "cpu")
        self.model = A2C(env=self.env, input_size=21).to(device)
        self.cplex = CPlexHandle(locator.cplex_log_folder, args.cplexpath, platform=args.platform)
        self.log = LogInfo()
        self.max_episodes = args.max_episodes
        self.max_steps = args.max_steps

    def env_step(self):
        # take matching step (Step 1 in paper)
        obs, pax_reward, done, info = self.env.pax_step(self.cplex)
        self.log.episode_reward += pax_reward
        # use GNN-RL policy (Step 2 in paper)
        action_rl = self.model.select_action(obs)
        # transform sample from Dirichlet into actual vehicle counts (i.e. (x1*x2*..*xn)*num_vehicles)
        desired_acc = {self.env.region[i]: int(action_rl[i] * dictsum(self.env.acc, self.env.time + 1)) for i in
                       range(len(self.env.region))}
        # solve minimum re-balancing distance problem (Step 3 in paper)
        acc_rl_tuple = [(n, int(round(desired_acc[n]))) for n in desired_acc]
        acc_tuple = [(n, int(self.env.acc[n][self.env.time + 1])) for n in self.env.acc]
        edge_attr = [(i, j, self.env.graph.get_edge_time(i, j)) for i, j in self.env.graph.get_all_edges()]

        reb_flow = self.cplex.solve_reb_flow(self.env.time, acc_rl_tuple, acc_tuple, edge_attr)
        reb_action = [reb_flow[i, j] for i, j in self.env.edges]

        # Take action in environment
        _, reb_reward, done, info = self.env.reb_step(reb_action)
        self.log.episode_reward += reb_reward
        # Store the transition in memory
        self.log.episode_served_demand += info.served_demand
        self.log.episode_reb_cost += info.reb_cost
        return done

    def train(self):
        # Initialize lists for logging
        epochs = trange(self.max_episodes)  # epoch iterator
        best_reward = -np.inf  # set best reward
        self.model.train()  # set model in train mode
        for episode in epochs:
            self.env.reset()  # initialize environment
            for step in range(self.max_steps):
                done = self.env_step()
                self.model.rewards.append(self.log.episode_reward)
                if done:
                    break
            self.model.training_step()
            epochs.set_description(self.log.get_desc(episode))
            if self.log.episode_reward >= best_reward:
                self.model.save_checkpoint(path=self.locator.save_best())
                best_reward = self.log.episode_reward
            self.log.append()
            self.model.log(self.log.to_obj('train'), path=self.locator.train_log())

    def test(self):
        self.model.load_checkpoint(path=self.locator.test_load())
        epochs = trange(self.max_episodes)  # epoch iterator
        # Initialize lists for logging
        for episode in epochs:
            self.env.reset()
            for step in range(self.max_steps):
                done = self.env_step()
                if done:
                    break
            # Send current statistics to screen
            epochs.set_description(self.log.get_desc(episode))
            # Log KPIs
            self.log.append()
            self.model.log(self.log.to_obj('test'), path=self.locator.test_log())
            break
