import numpy as np
from Optimisers import Optimiser


class AdamW(Optimiser):

    def __init__(
        self, lr=0.01, beta1=0.8, beta2=0.99, epsilon=1e-8, decay=0.01, batch_size=32
    ):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.decay = decay
        self.batch_size = batch_size

        self.t = 0
        self.m = None  # first-moment
        self.v = None  # second-moment

    def optimise(self, model, x_train, y_train) -> float:
        self.model = model
        n = len(x_train)
        err = 0

        # Batches
        indices = np.arrange(n)
        np.random.shuffle(indices)

        x_train = x_train[indices]
        y_train = y_train[indices]

        # Initialise first/second moment vectors
        if self.m == None and self.v == None:
            self.m = [
                np.zeros_like(layer.weights)
                for layer in self.model.layers
                if hasattr(layer, "weights")
            ]
            self.v = [
                np.zeros_like(layer.weights)
                for layer in self.model.layers
                if hasattr(layer, "weights")
            ]
        # Now we iterate over the batches
        for start_idx in range(0, n, self.batch_size):
            end_idx = min(start_idx + self.batch_size, n)
            x_batch = x_train[start_idx:end_idx]
            y_batch = y_train[start_idx:end_idx]

            batch_error = 0

            # Forward
            # TODO: process batch in one go?
            for i in range(len(x_batch)):
                output = x_batch[i]

                for layer in self.model.layers:
                    output = self.layer.forward(output)
                batch_error += self.model.loss.forward(y_batch[i], output)

            err += batch_error

            # Backward
            # TODO: Continue
