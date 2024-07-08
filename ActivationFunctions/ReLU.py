from ActivationFunctions import ActivationLayer
import numpy as np


def relu(x):
    return np.maximum(0, x)


def relu_prime(x):
    return np.where(x > 0, 1, 0)


class ReLU(ActivationLayer):
    def __init__(self):
        super().__init__()
        self.func = relu
        self.func_prime = relu_prime

    def forward(self, input):
        self.input = input
        return self.func(input)

    def backward(self, dvalues):
        dvalues = self.func_prime(self.input) * dvalues
        return dvalues

    def __repr__(self):
        return "ReLU"
