import numpy as np
from Layers import Layer

class FullyConnectedLayer(Layer):

    def __init__(self, input_size, output_size) -> None:
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5

    def forward(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output
    
    def backward(self, output_error):
        self.input_error = np.dot(output_error, self.weights.T)
        self.dweights = np.dot(self.input.T, output_error)
        self.dbias = output_error

        return self.input_error