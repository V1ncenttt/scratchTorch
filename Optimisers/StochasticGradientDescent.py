from Optimisers import Optimiser

class StochasticGradientDescent(Optimiser):

    def __init__(self, lr=0.01):
        self.lr = lr

    def optimise(self, model, x_train, y_train) -> float:
        
        self.model = model
        n = len(x_train)
        err = 0
        for i in range(n):
            output = x_train[i]

            for layer in self.model.layers:
            
                output = layer.forward(output)

            err += self.model.loss.forward(y_train[i], output)

            #Backward passs
            error = self.model.loss.backward(y_train[i], output)
            
            for layer in reversed(self.model.layers):
                error = layer.backward(error)
            
            for layer in self.model.layers:
                if hasattr(layer, 'weights'):
                    layer.update(self.lr)
                   

                    
        return err