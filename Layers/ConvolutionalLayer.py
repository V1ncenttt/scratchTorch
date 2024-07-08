import numpy as np
from Layers import Layer
import scipy.signal as signal


class Convolutional2DLayer(Layer):
    def __init__(self, kernel_size, input_depth, output_depth):
        self.input_depth = input_depth
        self.output_depth = output_depth
        self.kernel_size = kernel_size
        self.kernel_shape = (
            self.output_depth,
            self.input_depth,
            self.kernel_size,
            self.kernel_size,
        )
        self.weights = np.random.randn(*self.kernel_shape) * np.sqrt(
            2.0 / (self.input_depth * self.kernel_size * self.kernel_size)
        )
        self.bias = np.random.randn(self.output_depth)
        self.m_weights = None
        self.m_bias = None
        self.v_weights = None
        self.v_bias = None
        self.iterations = 0

    def get_output_shape(self, input_shape):
        batch_size, input_depth, input_height, input_width = input_shape
        output_height = input_height - self.kernel_size + 1
        output_width = input_width - self.kernel_size + 1
        return (self.output_depth, output_height, output_width)

    def forward(self, x):
        self.input = x

        batch_size, _, input_height, input_width = x.shape
        output_shape = self.get_output_shape(x.shape)
        self.output = np.zeros((batch_size, *output_shape))
        for n in range(batch_size):
            for i in range(self.output_depth):
                for j in range(self.input_depth):
                    self.output[n, i] += signal.correlate2d(
                        self.input[n, j], self.weights[i, j], "valid"
                    )
                self.output[n, i] += self.bias[i]
        return self.output

    def backward(self, error):
        batch_size = self.input.shape[0]
        self.dweights = np.zeros_like(self.weights)
        dinput = np.zeros_like(self.input)
        self.dbias = np.zeros_like(self.bias)
        for n in range(batch_size):
            for i in range(self.output_depth):
                self.dbias[i] += np.sum(error[n, i])
                for j in range(self.input_depth):
                    self.dweights[i, j] += signal.correlate2d(
                        self.input[n, j], error[n, i], "valid"
                    )
                    dinput[n, j] += signal.convolve2d(
                        error[n, i], self.weights[i, j], "full"
                    )
        return dinput
