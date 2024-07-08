import numpy as np
from LossFunctions import Loss


class CrossEntropy(Loss):
    def __init__(self):
        pass

    def forward(self, y_true, y_pred):
        return -np.sum(y_true * np.log(y_pred))

    def backward(self, y_true, y_pred):
        return y_pred - y_true
