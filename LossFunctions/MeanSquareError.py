import numpy as np
from LossFunctions import Loss


class MeanSquareError(Loss):
    def __init__(self):
        pass

    def forward(self, y_true, y_pred):
        return np.mean(np.square(y_pred - y_true))

    def backward(self, y_true, y_pred):
        return 2 * (y_pred - y_true) / y_pred.size
