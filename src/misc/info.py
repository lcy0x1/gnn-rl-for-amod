from numpy import mean


class StepInfo:

    def __init__(self, total_acc):
        self.total_acc = total_acc
        self.served_demand = 0
        self.operating_cost = 0
        self.revenue = 0
        self.reb_cost = 0
        self.pax_vehicle = 0
        self.reb_vehicle = 0
        self.idle_vehicle = 0

    def reset(self):
        self.served_demand = 0
        self.operating_cost = 0
        self.revenue = 0
        self.reb_cost = 0
        self.pax_vehicle = 0
        self.reb_vehicle = 0
        self.idle_vehicle = 0


def _format(i, rw, sd, rc, pv, rv, iv, pr):
    return f"Episode {i + 1} | Reward: {rw:.2f} | ServedDemand: {sd:.2f} | Reb. Cost: {rc:.2f} " \
           f"| Pax/Reb/Idle Vehicle: {pv * 100:.2f}%/{rv * 100:.2f}%/{iv * 100:.2f}% | PricePoint: {pr:.2f}"


class LogInfo:

    def __init__(self):
        self.reward = []
        self.served_demand = []
        self.reb_cost = []
        self.pax_vehicle = []
        self.reb_vehicle = []
        self.idle_vehicle = []
        self.price_point = []
        self.episode_reward = 0
        self.episode_served_demand = 0
        self.episode_reb_cost = 0
        self.episode_pax_vehicle = 0
        self.episode_reb_vehicle = 0
        self.episode_idle_vehicle = 0
        self.episode_price_point = 0

    def append(self):
        self.reward.append(self.episode_reward)
        self.served_demand.append(self.episode_served_demand)
        self.reb_cost.append(self.episode_reb_cost)
        self.pax_vehicle.append(self.episode_pax_vehicle)
        self.reb_vehicle.append(self.episode_reb_vehicle)
        self.idle_vehicle.append(self.episode_idle_vehicle)
        self.price_point.append(self.episode_price_point)
        self.episode_reward = 0
        self.episode_served_demand = 0
        self.episode_reb_cost = 0
        self.episode_pax_vehicle = 0
        self.episode_reb_vehicle = 0
        self.episode_idle_vehicle = 0
        self.episode_price_point = 0

    def to_obj(self, result_type: str):
        return {f'{result_type}_reward': self.reward,
                f'{result_type}_served_demand': self.served_demand,
                f'{result_type}_reb_cost': self.reb_cost,
                f'{result_type}_pax_vehicle': self.pax_vehicle,
                f'{result_type}_reb_vehicle': self.reb_vehicle,
                f'{result_type}_idle_vehicle': self.idle_vehicle,
                f'{result_type}_price_point': self.price_point}

    def from_obj(self, result_type: str, obj, last=-1):
        if last < 0:
            last = len(obj[f'{result_type}_reward'])
        self.reward = obj[f'{result_type}_reward'][0:last]
        self.served_demand = obj[f'{result_type}_served_demand'][0:last]
        self.reb_cost = obj[f'{result_type}_reb_cost'][0:last]
        self.pax_vehicle = obj[f'{result_type}_pax_vehicle'][0:last]
        self.reb_vehicle = obj[f'{result_type}_reb_vehicle'][0:last]
        self.idle_vehicle = obj[f'{result_type}_idle_vehicle'][0:last]
        self.price_point = obj[f'{result_type}_price_point'][0:last]

    def get_desc(self, episode):
        return _format(episode, self.episode_reward, self.episode_served_demand, self.episode_reb_cost,
                       self.episode_pax_vehicle, self.episode_reb_vehicle, self.episode_idle_vehicle,
                       self.episode_price_point)

    def get_average(self, episode):
        return _format(episode, mean(self.reward), mean(self.served_demand), mean(self.reb_cost),
                       mean(self.pax_vehicle), mean(self.reb_vehicle), mean(self.idle_vehicle),
                       mean(self.price_point))

    def accept(self, info: StepInfo):
        self.episode_served_demand += info.served_demand
        self.episode_reb_cost += info.reb_cost
        self.episode_pax_vehicle += info.pax_vehicle / info.total_acc
        self.episode_reb_vehicle += info.reb_vehicle / info.total_acc
        self.episode_idle_vehicle += info.idle_vehicle / info.total_acc
        self.episode_price_point += info.revenue / info.pax_vehicle

    def finish(self, step: int):
        self.episode_pax_vehicle /= step
        self.episode_reb_vehicle /= step
        self.episode_idle_vehicle /= step
        self.episode_price_point /= step
        pass
