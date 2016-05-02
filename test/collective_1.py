import numpy as np
import random

timespan = 2.0
numStep = 1000
numInvestors = 100

T = 10.0
mu = 0.04
sigma = 0.03
S0 = 10
alphapar = 0.2

investors = np.ones((numInvestors, numStep, 2))

def total_wealth(invests):
    return sum(invests)

def next_step(step, dt, intr, vol):
    er = random.gauss(0.0, 1.0)
    return step*np.exp((intr-0.5*vol**2)*dt + vol*er*np.sqrt(dt))


def calcPath(timespan, numStep, intr, vol, startPrice):

        step = startPrice
        prices = np.zeros(numStep)
        i = 0
        dt = timespan/numStep
        while i < numStep:
            step = next_step(step, dt, intr, vol)
            prices[i]=step
            investors[:,i,:]=step
            i += 1


        return prices

import matplotlib.pyplot as plt
plt.figure(1)
plt.subplot(211)
plt.plot(calcPath(T, 1000, mu, sigma, S0))

plt.subplot(212)
plt.plot(total_wealth(investors[:,:,0]))
plt.show()


plt.show()