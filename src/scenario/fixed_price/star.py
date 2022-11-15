from src.scenario.fixed_price.fixed_price_model import FixedPriceModelScenario
from src.scenario.scenario import Scenario


class Star2Complete(FixedPriceModelScenario):
    def __init__(self, n1=4, n2=4, sd=10, star_demand=20, complete_demand=1, star_center=[5, 6, 9, 10],
                 grid_travel_time=3, ninit=50, demand_ratio=[1, 1.5, 1.5, 1], alpha=0.2, fix_price=False):
        # beta - proportion of star network
        # alpha - parameter for uniform distribution of demand [1-alpha, 1+alpha]
        super(Star2Complete, self).__init__(n1=n1, n2=n2, sd=sd, ninit=ninit,
                                            grid_travel_time=grid_travel_time,
                                            fix_price=fix_price,
                                            alpha=alpha,
                                            demand_ratio=demand_ratio,
                                            demand_input={(i, j): complete_demand + (
                                                star_demand if i in star_center and j not in star_center else 0)
                                                          for i in range(0, n1 * n2) for j in range(0, n1 * n2) if
                                                          i != j}
                                            )
