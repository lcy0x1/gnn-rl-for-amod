import numpy as np

from src.scenario.scenario import Scenario


def evaluate_env(scenario: Scenario):
    trip = scenario.get_random_demand()
    demand = [0] * scenario.get_final_time()
    price = [0] * scenario.get_final_time()
    count = [0] * scenario.get_final_time()
    dist = [0] * scenario.get_final_time()
    max_price = 0
    for o, d, t, a, p in trip:
        if a == 0 or t < 0 or t >= scenario.get_final_time():
            continue
        demand[t] += a * scenario.get_demand_time(o, d, t)
        price[t] += a * p
        dist[t] += scenario.get_demand_time(o, d, t)
        count[t] += a
        max_price = max(max_price, p)
    n_price = np.array(price) / np.array(demand)
    n_dist = np.array(demand) / np.array(count)
    # display(n_price, './saved_files/scenario/price.png')
    # display(demand, './saved_files/scenario/demand.png')
    # display(n_dist, './saved_files/scenario/dist.png')
    print(scenario.get_init_acc(0) * scenario.get_graph().size())
    print(f'max_price: {max_price}')
