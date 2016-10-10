import numpy as np


class Stock:

    def __init__(self, id , price = 0.0):

        self.id = id
        self.price = price
        self.description = "Stock"
        self.author = "laputian"

class Holding:

    def __init__(self, security,  nr = 0):
        self.nr = nr
        self.security = security

    def value(self):
        return self.nr * self.security.price

    def sell(self, num):
        if num <= self.nr:
            self.nr = self.nr - num
            return num * self.security.price
        else:
            return 0.

    def buy(self, num):
            self.nr = self.nr + num
            return num * self.security.price


