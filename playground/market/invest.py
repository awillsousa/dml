import numpy as np

class Investor:

    def __init__(self, id, cash = 0.0, holdings=None):
        self.id = id
        self.holdings = holdings
        self.cash = cash
        self.description = "Investor"
        self.author = "laputian"

    def securities_value(self):
        sum = 0.
        for holding in self.holdings.itervalues():
            sum += holding.value()
        return sum

    def tot_assets_value(self):
        return self.securities_value() + self.cash

    def sell_sec(self, holding):
        self.cash += self.holdings[holding.security.id].sell(holding.nr)


    def buy_sec(self, holding):
        if self.cash >= holding.value():
            self.cash -= self.holdings[holding.security.id].buy(holding.nr)












