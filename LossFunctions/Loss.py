import numpy as np

class Loss:
    def __init__(self):
        pass

    def forward(self, y_pred, y_true):
        raise NotImplementedError

    def backward(self, y_pred, y_true):
        raise NotImplementedError