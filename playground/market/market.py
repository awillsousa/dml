import numpy as np
import random
from security import Stock

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
            return True
        else:
            return False


def create_market_securities(nr_of_stocks=1):
    d ={}
    for i in range(nr_of_stocks):
        d[str(i)] =  Stock(id = i, price = 1 + 10 * random.random())
    return d

def securities_tot_value(secs):
    sum = 0.0
    for value in secs.itervalues():
        sum += value.price
    return sum






