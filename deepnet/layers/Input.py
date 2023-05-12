import numpy as np


class Input:
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        return x

    def backward(self):
        pass
