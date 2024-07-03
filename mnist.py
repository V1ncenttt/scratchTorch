import numpy as np

from Networks import NeuralNetwork
from Layers import FullyConnectedLayer
from ActivationFunctions import Tanh, ReLU
from LossFunctions import MeanSquareError
from Optimisers import StochasticGradientDescent
import matplotlib.pyplot as plt

from keras.datasets import mnist
import keras.utils as np_utils

# load MNIST from server
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# training data : 60000 samples
# reshape and normalize input data
x_train = x_train.reshape(x_train.shape[0], 1, 28*28)
x_train = x_train.astype('float32')
x_train /= 255
# encode output which is a number in range [0,9] into a vector of size 10
# e.g. number 3 will become [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
y_train = np_utils.to_categorical(y_train)

# same for test data : 10000 samples
x_test = x_test.reshape(x_test.shape[0], 1, 28*28)
x_test = x_test.astype('float32')
x_test /= 255
y_test = np_utils.to_categorical(y_test)

# Example usage with MNIST dataset
net = NeuralNetwork()

net.add(FullyConnectedLayer(28*28, 100))                # input_shape=(1, 28*28)    ;   output_shape=(1, 100)
net.add(Tanh())
net.add(FullyConnectedLayer(100, 50))                   # input_shape=(1, 100)      ;   output_shape=(1, 50)
net.add(Tanh())
net.add(FullyConnectedLayer(50, 10))                    # input_shape=(1, 50)       ;   output_shape=(1, 10)
net.add(Tanh())

# train on 1000 samples
# as we didn't implemented mini-batch GD, training will be pretty slow if we update at each iteration on 60000 samples...
net.use_loss(MeanSquareError())
optimiser = StochasticGradientDescent(lr=0.1)
net.fit(x_train[0:1000], y_train[0:1000], optimiser, epochs=50)

# test on 3 samples
for i in range(10):
    print('------------------')
    out = net.predict(x_test[i])
    print("\n")
    print("predicted values : ")
    print(np.argmax(out), end="\n")
    print("true values : ")
    print(np.argmax(y_test[i]))

# plot first test sample
plt.imshow(x_test[0].reshape(28, 28), cmap='gray')
plt.show()