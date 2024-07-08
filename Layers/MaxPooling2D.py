import numpy as np
from Layers import Layer


class MaxPooling2DLayer(Layer):
    def __init__(self, pool_size, stride):
        self.pool_size = pool_size
        self.stride = stride

    def get_output_shape(self, input_shape):
        batch_size, input_depth, input_height, input_width = input_shape
        output_height = (input_height - self.pool_size) // self.stride + 1
        output_width = (input_width - self.pool_size) // self.stride + 1
        return (input_depth, output_height, output_width)

    def forward(self, x):
        self.input = x
        batch_size, input_depth, input_height, input_width = x.shape
        output_shape = self.get_output_shape(x.shape)
        self.output = np.zeros((batch_size, *output_shape))
        self.max_indices = np.zeros_like(self.output, dtype=int)

        for n in range(batch_size):
            for d in range(input_depth):
                for i in range(0, input_height - self.pool_size + 1, self.stride):
                    for j in range(0, input_width - self.pool_size + 1, self.stride):
                        region = self.input[
                            n, d, i : i + self.pool_size, j : j + self.pool_size
                        ]
                        max_val = np.max(region)
                        self.output[n, d, i // self.stride, j // self.stride] = max_val
                        self.max_indices[n, d, i // self.stride, j // self.stride] = (
                            np.argmax(region)
                        )
        return self.output

    def backward(self, error):
        batch_size, input_depth, input_height, input_width = self.input.shape
        dinput = np.zeros_like(self.input)

        for n in range(batch_size):
            for d in range(input_depth):
                for i in range(0, input_height - self.pool_size + 1, self.stride):
                    for j in range(0, input_width - self.pool_size + 1, self.stride):
                        max_index = self.max_indices[
                            n, d, i // self.stride, j // self.stride
                        ]
                        max_pos = np.unravel_index(
                            max_index, (self.pool_size, self.pool_size)
                        )
                        dinput[n, d, i + max_pos[0], j + max_pos[1]] = error[
                            n, d, i // self.stride, j // self.stride
                        ]
        return dinput
