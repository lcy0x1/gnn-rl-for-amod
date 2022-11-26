class ParameterGroup:

    def __init__(self, dist, step=1, cost=0.5, penalty=0.5, threshold=20, chargeback=4):
        self.dist = dist
        self.cost = cost * step
        self.penalty = penalty * step
        self.threshold = threshold
        self.chargeback = chargeback
