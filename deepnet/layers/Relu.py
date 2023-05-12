import numpy as np


class Relu:
    @staticmethod
    def forward(x):
        return (x > 0) * x

    def backward(self, x):
        return
