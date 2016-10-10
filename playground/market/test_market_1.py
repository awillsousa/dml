from security import Stock, Holding
from invest import Investor
from random import seed, randint, shuffle, sample
from market import trade
import matplotlib.pyplot as plt
import numpy as np
import itertools

#seed(4890)
secu = Stock(id='A', price= 10.)

def create_investors(nr = 10, cash_base=100., secu = secu):
    invs = {}
    for i in range(1, nr+1):
        holding = Holding(secu, nr=randint(30, 70))
        invs[(i)] = Investor(id=i, cash=cash_base * randint(1, 9), holdings = {holding.security.id: holding})
    return invs

investors = create_investors()
def calc_tot_assets():
    tot_assets = 0.
    for inv in investors.itervalues():
        tot_assets += inv.tot_assets_value()
    print 'Tot assets ' +str(tot_assets)

def calc_tot_cash():
    tot_cash = 0.
    for inv in investors.itervalues():
        tot_cash += inv.cash
    print 'Tot cash ' +str(tot_cash)

def trade_run(investors):
    for pair in itertools.combinations(sample(investors.values(), len(investors.values())), r=2):
            trade(pair[0], pair[1], Holding(secu, 1))
            # print ' 1 - ' + str(pair[0].id) + ' - ' + str(pair[0].cash) + ' - '  + str(pair[0].tot_assets_value())
            # print ' 2 - ' + str(pair[1].id) + ' - ' + str(pair[01].cash) + ' - ' + str(pair[1].tot_assets_value())

print("Init")
calc_tot_assets()
calc_tot_cash()

recs = []
a =[]
b= []
c = []
for inv in investors.values():
    a.append(int(inv.id))
    b.append(inv.tot_assets_value())
    c.append(inv.cash)
bp = plt.scatter(a,b, s=80, facecolors='none', edgecolors='y', label='Total value')
cp = plt.scatter(a,c, color='r', label='Initial cash')

for _ in range(10000):
    trade_run(investors)

print("Final")
calc_tot_assets()
calc_tot_cash()

x = []
z = []
for inv in investors.values():
    x.append(int(inv.id))
    z.append(inv.cash)
plt.scatter(x, z, color='g', label='Final cash')
zp = plt.xticks(np.arange(1, len(investors) + 1, 1.0))

plt.legend(loc='upper right', scatterpoints = 1)
plt.show()



