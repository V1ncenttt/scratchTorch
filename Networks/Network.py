import numpy as np
from Layers import Layer
from tabulate import tabulate

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
            headers = ["Layer", "Input Shape", "Output Shape", "Number of Parameters"]
            table = []
            total_params = 0

            for layer in self.layers:
                layer_name = type(layer).__name__


                if not getattr(layer, "input_shape", None) == None:
                    input_shape = getattr(layer, "input_shape", None)
                else:
                    input_shape = output_shape

                if not getattr(layer, "output_shape", None) == None:
                    output_shape = getattr(layer, "output_shape", None)
                else: 
                    output_shape = input_shape

                if hasattr(layer, "weights"):
                    num_params = np.prod(layer.weights.shape)
                    if hasattr(layer, "bias"):
                        num_params += np.prod(layer.bias.shape)
                    total_params += num_params
                else:
                    num_params = 0

                table.append([layer_name, input_shape, output_shape, num_params])

            representation = "=============================\n"
            representation += "Neural Network (ScratchTorch)\n"
            representation += "=============================\n"
            representation += tabulate(table, headers, tablefmt="grid")
            representation += f"\n\nTotal number of trainable parameters: {total_params}\n"
            
            return representation
