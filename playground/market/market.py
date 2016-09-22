import numpy as np

class Market:

    def __init__(self, id = 'Market', investors=[]):

        self.id = id
        self.investors= investors
        self.description = "Market"
        self.author = "laputian"

    def tot_stock_value(self):
        sum = 0.0
        for investor in self.investors:
            for holding in investor.holdings.itervalues():
                sum += holding.value()
        return sum

    def tot_cash(self):
        sum = 0.0
        for investor in self.investors:
                sum += investor.cash
        return sum

def trade(seller, buyer, trade_props):
        if (seller.holdings[trade_props.security.id].nr >= trade_props.nr and
                    buyer.cash >= trade_props.value()):
            buyer.buy_sec(trade_props)
            seller.sell_sec(trade_props)

