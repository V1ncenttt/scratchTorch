from Optimisers import Optimiser

import numpy as np

class MiniBatchGradientDescent(Optimiser):

    def __init__(self, lr=0.01, batch_size=32):
        self.lr = lr
        self.batch_size = batch_size

    def optimise(self, model, x_train, y_train) -> float:
        
        self.model = model
        n = len(x_train)
        err = 0
        
        # Shuffle the training data
        indices = np.arange(n)
        np.random.shuffle(indices)
        
        x_train = x_train[indices]
        y_train = y_train[indices]
        
        # Iterate over mini-batches
        for start_idx in range(0, n, self.batch_size):
            end_idx = min(start_idx + self.batch_size, n)
            x_batch = x_train[start_idx:end_idx]
            y_batch = y_train[start_idx:end_idx]
            
            batch_err = 0
            
            # Forward pass for the mini-batch
            for i in range(len(x_batch)):
                output = x_batch[i]
                
                for layer in self.model.layers:
                    output = layer.forward(output)
                    
                batch_err += self.model.loss.forward(y_batch[i], output)
            
            err += batch_err
            
            # Backward pass for the mini-batch
            for i in range(len(x_batch)):
                output = x_batch[i]
                
                for layer in self.model.layers:
                    output = layer.forward(output)
                    
                error = self.model.loss.backward(y_batch[i], output)
                
                for layer in reversed(self.model.layers):
                    error = layer.backward(error)
            
            # Update parameters after the mini-batch
            for layer in self.model.layers:
                if hasattr(layer, 'weights'):
                    layer.update(self.lr)
                    
        return err 