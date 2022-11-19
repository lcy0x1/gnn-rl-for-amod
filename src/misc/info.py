from collections import defaultdict
from enum import Enum

from numpy import mean

LogEntry = Enum('LogEntry', ['value_loss', 'policy_loss', 'gradient', 'reward', 'revenue',
                             'served_demand', 'missed_demand',
                             'reb_cost', 'price_point', 'pax_vehicle', 'reb_vehicle', 'idle_vehicle'])


class StepInfo:

    def __init__(self, total_acc):
        self.total_acc = total_acc
        self.served_demand = 0
        self.missed_demand = 0
        self.operating_cost = 0
        self.revenue = 0
        self.reb_cost = 0
        self.pax_vehicle = 0
        self.reb_vehicle = 0
        self.idle_vehicle = 0

    def reset(self):
        self.served_demand = 0
        self.missed_demand = 0
        self.operating_cost = 0
        self.revenue = 0
        self.reb_cost = 0
        self.pax_vehicle = 0
        self.reb_vehicle = 0
        self.idle_vehicle = 0


def _format(i, d):
    gr = d(LogEntry.gradient)
    pl = d(LogEntry.policy_loss)
    vl = d(LogEntry.value_loss)
    rw = d(LogEntry.reward)
    rn = d(LogEntry.revenue)
    rc = d(LogEntry.reb_cost)
    sd = d(LogEntry.served_demand)
    md = d(LogEntry.missed_demand)
    pr = d(LogEntry.price_point)
    pv = d(LogEntry.pax_vehicle)
    rv = d(LogEntry.reb_vehicle)
    iv = d(LogEntry.idle_vehicle)

    sdr = sd / (sd + md) * 100
    return f"Episode {i + 1} | " \
           f"Policy Loss: {pl:.2f} | " \
           f"Value Loss: {vl:.2f} | " \
           f"Actor Reward: {gr:.2f} | " \
           f"Reward: {rw:.2f} | " \
           f"Revenue: {rn:.2f} | " \
           f"ServedDemand: {sdr:.2f}% | " \
           f"Reb. Cost: {rc:.2f} | " \
           f"Pax/Reb/Idle Vehicle: {pv * 100:.2f}%/{rv * 100:.2f}%/{iv * 100:.2f}% | " \
           f"PricePoint: {pr:.2f}"


class LogInfo:

    def __init__(self):
        self.lists = defaultdict(lambda: [])
        self.episode_data = defaultdict(float)

    def append(self):
        for e in LogEntry:
            self.lists[e].append(self.episode_data[e])
            self.episode_data[e] = 0

    def to_obj(self, result_type: str):
        ans = dict()
        for e in LogEntry:
            ans[f'{result_type}_{e.name}'] = self.lists[e]
        return ans

    def from_obj(self, result_type: str, obj, last=-1):
        if last < 0:
            last = len(obj[f'{result_type}_reward'])
        for e in LogEntry:
            self.lists[e] = obj[f'{result_type}_{e.name}'][0:last]

    def get_desc(self, episode):
        return _format(episode, lambda e: self.episode_data[e])

    def get_average(self, episode):
        return _format(episode, lambda e: mean(self.lists[e]))

    def accept(self, info: StepInfo):
        self.episode_data[LogEntry.served_demand] += info.served_demand
        self.episode_data[LogEntry.missed_demand] += info.missed_demand
        self.episode_data[LogEntry.revenue] += info.revenue
        self.episode_data[LogEntry.reb_cost] += info.reb_cost
        self.episode_data[LogEntry.pax_vehicle] += info.pax_vehicle / info.total_acc
        self.episode_data[LogEntry.reb_vehicle] += info.reb_vehicle / info.total_acc
        self.episode_data[LogEntry.idle_vehicle] += info.idle_vehicle / info.total_acc
        self.episode_data[LogEntry.price_point] += 0 if info.pax_vehicle == 0 else info.revenue / info.pax_vehicle

    def finish(self, step: int):
        self.episode_data[LogEntry.pax_vehicle] /= step
        self.episode_data[LogEntry.reb_vehicle] /= step
        self.episode_data[LogEntry.idle_vehicle] /= step
        self.episode_data[LogEntry.price_point] /= step

    def get_reward(self):
        return self.episode_data[LogEntry.reward]

    def add_reward(self, r):
        self.episode_data[LogEntry.reward] += r
