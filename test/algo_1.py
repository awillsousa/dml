import numpy as np
import random

T = 2.0
mu = 0.02
sigma = 0.01
S0 = 10


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
            i += 1


        return prices

import matplotlib.pyplot as plt
plt.plot(calcPath(T, 1000, mu, sigma, S0))
plt.show()




