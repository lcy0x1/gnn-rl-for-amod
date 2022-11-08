import os
import subprocess
from collections import defaultdict

from src.misc.utils import mat2str


def check(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


class CPlexHandle:

    def __init__(self, res_path: str, cplexpath="/opt/ibm/ILOG/CPLEX_Studio128/opl/bin/x86-64_linux/",
                 platform='linux'):
        self.cplex_path = cplexpath
        self.platform = platform
        self.mod_path = os.getcwd().replace('\\', '/') + '/src/cplex_mod/'
        self.reb_path = os.getcwd().replace('\\', '/') + '/saved_files/cplex_logs/rebalancing/' + res_path + '/'
        self.mat_path = os.getcwd().replace('\\', '/') + '/saved_files/cplex_logs/matching/' + res_path + '/'
        check(self.reb_path)
        check(self.mat_path)
        self.my_env = os.environ.copy()
        if self.platform == 'mac':
            self.my_env["DYLD_LIBRARY_PATH"] = self.cplex_path
        else:
            self.my_env["LD_LIBRARY_PATH"] = self.cplex_path

    def run(self, out_file, modfile, datafile):
        with open(out_file, 'w') as output_f:
            subprocess.check_call([self.cplex_path + "oplrun", modfile, datafile], stdout=output_f, env=self.my_env)
        output_f.close()

    def solve_reb_flow(self, t: int, acc_rl_tuple, acc_tuple, edge_attr):
        datafile = self.reb_path + f'data_{t}.dat'
        resfile = self.reb_path + f'res_{t}.dat'
        with open(datafile, 'w') as file:
            file.write('path="' + resfile + '";\r\n')
            file.write('edgeAttr=' + mat2str(edge_attr) + ';\r\n')
            file.write('accInitTuple=' + mat2str(acc_tuple) + ';\r\n')
            file.write('accRLTuple=' + mat2str(acc_rl_tuple) + ';\r\n')
        modfile = self.mod_path + 'minRebDistRebOnly.mod'

        out_file = self.reb_path + f'out_{t}.dat'

        self.run(out_file, modfile, datafile)

        # 3. collect results from file
        flow = defaultdict(float)
        with open(resfile, 'r', encoding="utf8") as file:
            for row in file:
                item = row.strip().strip(';').split('=')
                if item[0] == 'flow':
                    values = item[1].strip(')]').strip('[(').split(')(')
                    for v in values:
                        if len(v) == 0:
                            continue
                        i, j, f = v.split(',')
                        flow[int(i), int(j)] = float(f)
        return flow

    def solve_mat_flow(self, t: int, demand_attr, acc_tuple):
        datafile = self.mat_path + f'data_{t}.dat'
        resfile = self.mat_path + f'res_{t}.dat'
        out_file = self.mat_path + f'out_{t}.dat'
        with open(datafile, 'w') as file:
            file.write('path="' + resfile + '";\r\n')
            file.write('demandAttr=' + mat2str(demand_attr) + ';\r\n')
            file.write('accInitTuple=' + mat2str(acc_tuple) + ';\r\n')
        modfile = self.mod_path + 'matching.mod'

        self.run(out_file, modfile, datafile)

        flow = defaultdict(float)
        with open(resfile, 'r', encoding="utf8") as file:
            for row in file:
                item = row.replace('e)', ')').strip().strip(';').split('=')
                if item[0] == 'flow':
                    values = item[1].strip(')]').strip('[(').split(')(')
                    for v in values:
                        if len(v) == 0:
                            continue
                        i, j, f = v.split(',')
                        flow[int(i), int(j)] = float(f)
        return flow
