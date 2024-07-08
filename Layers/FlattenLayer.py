from Layers import Layer


class FlattenLayer(Layer):
    def forward(self, x):
        self.input_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, error):
        return error.reshape(self.input_shape)
