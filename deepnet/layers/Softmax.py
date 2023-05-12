import numpy as np


class Softmax:
    @staticmethod
    def forward(x):
        temp = np.exp(x)
        return temp / np.sum(temp)

    @staticmethod
    def backward(x, y):
        temp = (x - y)
        return temp / len(y)
