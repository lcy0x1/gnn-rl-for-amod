class ResourceLocator:

    def __init__(self, directory, suffix):
        self.directory = directory
        self.env_json_file = "data/scenario_nyc4x4.json"
        self.cplex_log_folder = f'scenario_nyc4_{suffix}'
        self.ckpt_path = 'ckpt/nyc4'
        self.rllog_path = 'rl_logs/nyc4'
        self.target_pth = f'a2c_gnn_test_{suffix}.pth'
        self.test_pth = f'a2c_gnn_eval_{suffix}.pth'
        self.graph_folder = f'graphs/nyc4_{suffix}'

    def pre_train(self, path):
        return f"./{self.directory}/{self.ckpt_path}/a2c_gnn_test_{path}.pth"

    def save_best(self):
        return f"./{self.directory}/{self.ckpt_path}/{self.target_pth}"

    def train_log(self):
        return f"./{self.directory}/{self.rllog_path}/{self.target_pth}"

    def test_load(self):
        return f"./{self.directory}/{self.ckpt_path}/{self.target_pth}"

    def test_log(self):
        return f"./{self.directory}/{self.rllog_path}/{self.test_pth}"

    def save_graphs(self, source: str):
        return f"./{self.directory}/{self.graph_folder}_{source}/"
