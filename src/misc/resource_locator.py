class ResourceLocator:

    def __init__(self, directory, suffix):
        self.directory = directory
        self.env_json_file = "data/scenario_nyc4x4.json"
        self.cplex_log_folder = 'scenario_nyc4'
        self.ckpt_path = 'ckpt/nyc4'
        self.rllog_path = 'rl_logs/nyc4'
        self.target_pth = f'a2c_gnn_test_{suffix}.pth'
        self.saved_pth = f'a2c_gnn_{suffix}.pth'
        self.graph_folder = f'graphs/nyc4_{suffix}'

    def save_best(self):
        return f"./{self.directory}/{self.ckpt_path}/{self.target_pth}"

    def train_log(self):
        return f"./{self.directory}/{self.rllog_path}/{self.target_pth}"

    def test_load(self):
        return f"./{self.directory}/{self.ckpt_path}/{self.saved_pth}"

    def test_log(self):
        return f"./{self.directory}/{self.rllog_path}/{self.target_pth}"

    def save_graphs(self):
        return f"./{self.directory}/{self.graph_folder}/"
