from __future__ import print_function
import argparse

from src.envs.trainer import Trainer
from src.misc.resource_locator import ResourceLocator
from src.misc.display import view
from src.scenario.evaluator import evaluate_env

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A2C-GNN')

    # Simulator parameters
    parser.add_argument('--seed', type=int, default=12345, metavar='S',
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
    parser.add_argument('--view', type=str, default='none',
                        help='view results: none for nothing, train for training results. '
                             'scenario for environment inspection (default: none)')
    parser.add_argument('--test', type=bool, default=False,
                        help='activates test mode for agent evaluation')
    parser.add_argument('--cplexpath', type=str, default='/Applications/CPLEX_Studio221/opl/bin/x86-64_osx/',
                        help='defines directory of the CPLEX installation')
    parser.add_argument('--platform', type=str, default='mac',
                        help='operating system. windows/linux/mac')
    parser.add_argument('--directory', type=str, default='saved_files',
                        help='defines directory where to save files')
    parser.add_argument('--max_episodes', type=int, default=20000, metavar='N',
                        help='number of episodes to train agent (default: 16k)')
    parser.add_argument('--max_steps', type=int, default=60, metavar='N',
                        help='number of steps per episode (default: T=60)')
    parser.add_argument('--no-cuda', type=bool, default=True,
                        help='disables CUDA training')

    # Network parameters
    parser.add_argument('--vehicle_forecast', type=int, default=10,
                        help='time steps for the network to preview vehicle availability')
    parser.add_argument('--demand_forecast', type=int, default=10,
                        help='time steps for the network to preview demand')
    parser.add_argument('--instance_suffix', type=str, default='default',
                        help='name for instance, helps to separate data')
    parser.add_argument('--fixed_price', type=bool, default=False,
                        help='make the environment to ignore price input')
    parser.add_argument('--resume', type=bool, default=False,
                        help='start from previous place')

    args = parser.parse_args()

    locator = ResourceLocator(args.directory, args.instance_suffix)
    if args.view == 'train':
        view(locator)
    elif args.view == 'scenario':
        trainer = Trainer(args, locator)
        evaluate_env(trainer.scenario)
    else:
        trainer = Trainer(args, locator)
        if not args.test:
            trainer.train()
        else:
            trainer.test()
