# http://stackoverflow.com/questions/13202799/python-code-geometric-brownian-motion-whats-wrong
import matplotlib.pyplot as plt
import numpy as np

T = 2
mu = 0.1
sigma = 0.1
S0 = 10
dt = 0.01
alpha = 0.1
N = round(T/dt)
t = np.linspace(0, T, N)
W = np.random.standard_normal(size = N)
W = np.cumsum(W)*np.sqrt(dt) ### standard brownian motion ###
X = ((mu + alpha * W)-0.5*sigma**2)*t + sigma*W
S = S0*np.exp(X) ### geometric brownian motion ###
plt.plot(t, S, '-', t, W, '--')
plt.show()