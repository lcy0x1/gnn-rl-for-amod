class RunningAverage:

    def __init__(self, rate=0.95):
        self.rate = rate
        self.best = 0
        self.current = 0

    def accept(self, val):
        self.current = self.current * self.rate + val
        if self.current > self.best:
            self.best = self.current
            return True
        return False
