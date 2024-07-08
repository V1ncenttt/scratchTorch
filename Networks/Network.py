import numpy as np
from Layers import Layer


class NeuralNetwork:

    def __init__(self) -> None:
        self.layers = []
        self.loss = None

    def add(self, layer: Layer):
        self.layers.append(layer)

    def use_loss(self, loss):
        self.loss = loss

    def forward(self, input_data):

        n_inputs = len(input_data)
        result = []

        for i in range(n_inputs):
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward(output)
            result.append(output)

        return result

    def predict(self, input_data):
        # sample dimension first
        samples = len(input_data)
        result = []

        # run network over all samples
        for i in range(samples):
            # forward propagation
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward(output)
            result.append(output)

        return result

    def fit(self, x_train, y_train, optimiser, epochs):

        n = len(x_train)

        for i in range(epochs):
            err = optimiser.optimise(self, x_train, y_train)
            # calculate average error on all samples
            err /= n
            print("epoch %d/%d   error=%f" % (i + 1, epochs, err))

    def __repr__(self) -> str:
        representation = ""

        for layer in self.layers:
            if hasattr(layer, "weights"):
                representation += (
                    str(layer)
                    + " - Weights: "
                    + str(layer.weights)
                    + " - Bias: "
                    + str(layer.bias)
                    + "\n"
                )
            else:
                representation += str(layer) + "\n"
        return representation
