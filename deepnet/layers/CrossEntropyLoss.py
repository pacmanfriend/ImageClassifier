import numpy as np


class CrossEntropyLoss:
    @staticmethod
    def forward(target, predict):
        return -target * np.log(predict)

    @staticmethod
    def backprop(target, predict):
        return -(target / predict)
