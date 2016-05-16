import numpy as np
import random

random.seed([1])

numSteps = 1000
numInvestors = 100

investors = np.ones((numInvestors, numSteps))

def weightedBoolProb(minimum):
    if random.random()>minimum:
        return True
    return False

def total_wealth(invests):
    return sum(invests)


min = 0.5

for i in range(numSteps):
    buy=0
    for j in range(numInvestors):
        for k in range(j+1, numInvestors):
            if weightedBoolProb(min) and not weightedBoolProb(min) and investors[j,i] > 0.0 and investors[k,i] < 2.0:
                investors[j,i] -= 1.0
                investors[k,i] += 1.0
                buy += 1
            elif weightedBoolProb(min) and not weightedBoolProb(min) and investors[k, i] > 0.0 and investors[j, i] < 2.0:
                investors[j, i] += 1.0
                investors[k, i] -= 1.0
                buy += 1



