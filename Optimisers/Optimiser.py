class Optimiser:
    def __init__(self, model):
        self.model = model

    def optimise(self, x_train, y_train) -> float:
        raise NotImplementedError
