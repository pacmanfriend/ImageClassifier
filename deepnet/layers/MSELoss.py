import numpy as np


class MSELoss:
    @staticmethod
    def forward(target, predict):
        return (target - predict) ** 2

    @staticmethod
    def backprop(target, predict):
        return predict - target
