import numpy as np


class Sigmoid:
    @staticmethod
    def forward(x):
        y = 1 / (1 + np.exp(-x))
        return y

    @staticmethod
    def backward(x):
        return x * (1 - x)
