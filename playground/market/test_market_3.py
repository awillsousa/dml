from security import Stock, Holding
from invest import Investor
from random import seed, randint, shuffle, sample
from market import trade
import matplotlib.pyplot as plt
import numpy as np
import math
import itertools
import matplotlib.mlab as mlab

#seed(4890)
init_price = 75.
secu = Stock(id='A', price= init_price)

def create_investors(nr = 10, cash_base=80., secu = secu):
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


def trade_run(investors, init_price = init_price,  runs = 1000):
    plot_arr = [init_price]
    for _ in range(runs):
        for pair in itertools.combinations(sample(investors.values(), len(investors.values())), r=2):
                success = trade(pair[0], pair[1], Holding(secu, 1))
                # if success:
                #     secu.price +=  secu.price/100
                # else:trade_run(investors, runs = 1000)
                #     secu.price += -secu.price/200
                if success:
                    secu.price += 0.1
                else:
                    secu.price = max(0.0, secu.price - 0.1)
        plot_arr.append(secu.price)
    return plot_arr
            # print ' 1 - ' + str(pair[0].id) + ' - ' + str(pair[0].cash) + ' - '  + str(pair[0].tot_assets_value())
            # print ' 2 - ' + str(pair[1].id) + ' - ' + str(pair[01].cash) + ' - ' + str(pair[1].tot_assets_value())

def gauss(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

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
bp = plt.scatter(a,b, s=80, facecolors='none', edgecolors='r', label='Total value initial')
cp = plt.scatter(a,c, color='r', label='Initial cash')

plot_arr = trade_run(investors, runs = 10000)

print("Final")
calc_tot_assets()
calc_tot_cash()

x = []
y = []
z = []
for inv in investors.values():
    x.append(int(inv.id))
    y.append(inv.tot_assets_value())
    z.append(inv.cash)
plt.scatter(x, y, s=80, facecolors='none', edgecolors='g', label='Total value final')
plt.scatter(x, z, color='g', label='Final cash')
zp = plt.xticks(np.arange(1, len(investors) + 1, 1.0))

plt.legend(loc = 4 ,scatterpoints = 1, prop={'size':10})
plt.show()
plt.plot(plot_arr)
plt.show()

from scipy.stats import kurtosistest

plot_diffs = np.asarray(plot_arr)
plot_diffs = np.roll(plot_diffs, 1) - plot_diffs
plot_diffs[0] = 0.
count, bins, ignored = plt.hist(plot_diffs, 30, normed=True)
min = plot_diffs.min()
max = plot_diffs.max()
interval = np.linspace(min, max, 100)
mu = np.mean(plot_diffs)
variance = np.var(plot_diffs)
sigma = np.sqrt(variance)
plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
             np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
        linewidth=2, color='r')
plt.title("Gaussian Histogram")
plt.xlabel("Kurtosis test p-value = "+str(kurtosistest(plot_diffs)[1]))
plt.ylabel("Frequency")

fig = plt.gcf()

plt.show()



