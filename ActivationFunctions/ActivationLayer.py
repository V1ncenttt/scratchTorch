from Layers import Layer


class ActivationLayer(Layer):

    def __init__(self) -> None:
        pass

    def forward(self, input_data):
        raise NotImplementedError

    def backward(self, output_error):
        raise NotImplementedError
