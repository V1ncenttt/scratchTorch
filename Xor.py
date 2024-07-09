import numpy as np

from Networks import NeuralNetwork
from Optimisers import StochasticGradientDescent
from Layers import FullyConnectedLayer
from ActivationFunctions import ReLU, Tanh
from LossFunctions import MeanSquareError, CrossEntropy

# training data
x_train = np.array([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

# network

net = NeuralNetwork()
net.add(FullyConnectedLayer(2, 3))
net.add(Tanh())
net.add(FullyConnectedLayer(3, 1))
net.add(Tanh())
optimiser = StochasticGradientDescent(lr=0.1)
# train
net.use_loss(CrossEntropy())
print(net)
exit()
net.fit(x_train, y_train, optimiser, epochs=1000)

# test
out = net.predict(x_train)
print(out)
