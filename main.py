from __future__ import print_function
import argparse
import torch

from src.algos.cplex_handle import CPlexHandle
from src.envs.amod_env import AMoD
from src.envs.trainer import Trainer
from src.scenario.json_raw_data import JsonRawDataScenario
from src.algos.a2c_gnn import A2C
from src.misc.display import display


def view(directory):
    path = f"./{directory}/graphs/nyc4-refactored/"
    logs = torch.load(f"./{directory}/rl_logs/nyc4/a2c_gnn_test.pth")
    print(len(logs['train_reward']))
    display(logs['train_reward'], f"{path}a2c_gnn_train_reward.png")
    display(logs['train_served_demand'], f"{path}a2c_gnn_train_served_demand.png")
    display(logs['train_reb_cost'], f"{path}a2c_gnn_train_reb_cost.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A2C-GNN')

    # Simulator parameters
    parser.add_argument('--seed', type=int, default=10, metavar='S',
                        help='random seed (default: 10)')
    parser.add_argument('--demand_ratio', type=int, default=0.5, metavar='S',
                        help='demand_ratio (default: 0.5)')
    parser.add_argument('--json_hr', type=int, default=7, metavar='S',
                        help='json_hr (default: 7)')
    parser.add_argument('--json_tsetp', type=int, default=3, metavar='S',
                        help='minutes per timestep (default: 3min)')
    parser.add_argument('--beta', type=int, default=0.5, metavar='S',
                        help='cost of rebalancing (default: 0.5)')

    # Model parameters
    parser.add_argument('--view', type=bool, default=False,
                        help='view results')
    parser.add_argument('--test', type=bool, default=False,
                        help='activates test mode for agent evaluation')
    parser.add_argument('--cplexpath', type=str, default='/Applications/CPLEX_Studio221/opl/bin/x86-64_osx/',
                        help='defines directory of the CPLEX installation')
    parser.add_argument('--platform', type=str, default='mac',
                        help='operating system. windows/linux/mac')
    parser.add_argument('--directory', type=str, default='saved_files',
                        help='defines directory where to save files')
    parser.add_argument('--max_episodes', type=int, default=16000, metavar='N',
                        help='number of episodes to train agent (default: 16k)')
    parser.add_argument('--max_steps', type=int, default=60, metavar='N',
                        help='number of steps per episode (default: T=60)')
    parser.add_argument('--no-cuda', type=bool, default=True,
                        help='disables CUDA training')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")

    if args.view:
        view(args.directory)
    else:
        scenario = JsonRawDataScenario(json_file="data/scenario_nyc4x4.json", sd=args.seed,
                                       demand_ratio=args.demand_ratio,
                                       json_hr=args.json_hr, json_tstep=args.json_tsetp)
        env = AMoD(scenario, beta=args.beta)
        model = A2C(env=env, input_size=21).to(device)
        cplex_handle = CPlexHandle('scenario_nyc4', args.cplexpath, platform=args.platform)
        trainer = Trainer(args, model, env, cplex_handle)
        if not args.test:
            trainer.train()
        else:
            trainer.test()
