from ActivationFunctions import ActivationLayer
import numpy as np


def tanh(x):
    return np.tanh(x)


def tanh_prime(x):
    return 1 - np.tanh(x) ** 2


class Tanh(ActivationLayer):
    def __init__(self):
        super().__init__()
        self.func = tanh
        self.func_prime = tanh_prime

    def forward(self, input):
        self.input = input
        return self.func(input)

    def backward(self, dvalues):
        dvalues = self.func_prime(self.input) * dvalues
        return dvalues

    def __repr__(self):
        return "TanH"
