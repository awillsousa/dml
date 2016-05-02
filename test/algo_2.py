import numpy as np
import random

T = 2.0
mu = 0.02
sigma = 0.01
S0 = 10
alphapar = 0.2


def next_step(step, dt, intr, vol, alpha, old_er):
    er = random.gauss(0.0, 1.0)
    next = step*np.exp(((intr+alpha*old_er)-0.5*vol**2)*dt + vol*er*np.sqrt(dt))
    return next, er-old_er


def calcPath(timespan, numStep, intr, vol, alpha, startPrice):

    step = startPrice
    prices = np.zeros(numStep)
    i = 0
    old_er = 0.0
    dt = timespan/numStep
    while i < numStep:
        step, old_er = next_step(step, dt, intr, vol, alpha,old_er)
        prices[i]=step
        i += 1

    return prices

def trend(prices, last, interval):

    if last < interval:
        return 0

    return sum(prices[last-interval:last + 1 ])/interval


import matplotlib.pyplot as plt
plt.plot(calcPath(T, 1000, mu, sigma, alphapar,S0))
plt.show()




