class StepInfo:

    def __init__(self):
        self.served_demand = 0
        self.operating_cost = 0
        self.revenue = 0
        self.reb_cost = 0

    def reset(self):
        self.served_demand = 0
        self.operating_cost = 0
        self.revenue = 0
        self.reb_cost = 0


class LogInfo:

    def __init__(self):
        self.reward = []
        self.served_demand = []
        self.reb_cost = []
        self.episode_reward = 0
        self.episode_served_demand = 0
        self.episode_reb_cost = 0

    def to_obj(self, result_type: str):
        return {f'{result_type}_reward': self.reward,
                f'{result_type}_served_demand': self.served_demand,
                f'{result_type}_reb_cost': self.reb_cost}

    def get_desc(self, episode):
        return f"Episode {episode + 1} | Reward: {self.episode_reward:.2f} | ServedDemand: {self.episode_served_demand:.2f} | Reb. Cost: {self.episode_reb_cost}"

    def append(self):
        self.reward.append(self.episode_reward)
        self.served_demand.append(self.episode_served_demand)
        self.reb_cost.append(self.episode_reb_cost)
        self.episode_reward = 0
        self.episode_served_demand = 0
        self.episode_reb_cost = 0
