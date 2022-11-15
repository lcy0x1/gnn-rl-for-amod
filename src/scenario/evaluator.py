from collections import defaultdict

from src.misc.display import display
from src.scenario.scenario import Scenario


def evaluate_env(scenario: Scenario):
    trip = scenario.get_random_demand()
    total = [0] * (scenario.get_final_time() * 2)
    avg_price = 0
    count = 0
    for o, d, t, a, p in trip:
        if a == 0:
            continue
        total[t] += a * p
        avg_price += a * p
        count += a * scenario.get_demand_time(o, d, t)
    print(f'Average price: {avg_price / count}')
    display(total, './demand.png')
