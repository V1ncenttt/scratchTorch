class Layer:
    def __init__(self) -> None:
        self.input = None
        self.output.shape = self.input.shape()

    def forward(self, input):
        raise NotImplementedError

    def backward(self, output_error):
        raise NotImplementedError
