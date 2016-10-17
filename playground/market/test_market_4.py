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
init_price = 85.
tot_runs = 1000
secu = Stock(id='A', price= init_price)

def create_investors(nr = 20, cash_base=300., secu = secu):
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

def trade_run(investors, num_sell =1 ):
    for pair in itertools.combinations(sample(investors.values(), len(investors.values())), r=2):
            success = trade(pair[0], pair[1], Holding(secu, num_sell) )
            if success:
                secu.price +=  secu.price/1000
            else:
                secu.price += -secu.price/1000
    return secu.price, success

def trade_run_iterate(investors, init_price = init_price, runs = tot_runs):
    plot_arr = [init_price]
    num_sell = 1
    for _ in range(runs):
        tr = trade_run(investors, num_sell)
        plot_arr.append(tr[0])
        if tr[1]:
            num_sell = 1
        else:
            num_sell = 2
    return plot_arr

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

plot_arr = trade_run_iterate(investors, runs = tot_runs)

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
count, bins, ignored = plt.hist(plot_diffs, 100, normed=True)
min = plot_diffs.min()
max = plot_diffs.max()
mu = np.mean(plot_diffs)
variance = np.var(plot_diffs)
sigma = np.sqrt(variance)
plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
             np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
        linewidth=2, color='r')
plt.title("Mean = "+str(mu) + " , std ="+str(sigma))
plt.xlabel("Kurtosis test p-value = "+str(kurtosistest(plot_diffs)[1]))
plt.ylabel("Frequency")

fig = plt.gcf()

plt.show()



