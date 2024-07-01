from Layers import Layer

class ActivationLayer(Layer):

    def __init__(self, activation, activation_prime) -> None:
        self.activation = activation
        self.activation_prime = activation_prime
    
    def forward(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output
    
    def backward(self, output_error):
        return self.activation_prime(self.input) * output_error