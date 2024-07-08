# scratchTorch

## Table of Contents

- [About](#about)
- [Getting Started](#getting_started)
- [Usage](#usage)
- [Contributing](../CONTRIBUTING.md)

## About <a name = "about"></a>

scratchTorch is my personal project aimed at reimplementing PyTorch from scratch using Python. The goal is to gain a deep understanding of how deep learning frameworks work by building one myself. This project includes basic neural network components such as convolutional layers, pooling layers, and backpropagation mechanisms.

## Getting Started <a name = "getting_started"></a>

Follow these instructions to get a copy of scratchTorch up and running on your local machine for development and testing purposes.

### Prerequisites

You'll need the following software to run scratchTorch:

- Python 3.x
- NumPy
- SciPy

To install the necessary packages, you can use pip:


```bash
pip install numpy scipy
```

### Installing

To install simply clone the repository

```bash
git clone https://github.com/V1ncenttt/scratchTorch.git
cd scratchTorch
```

You can optionally use a virtual environment.

```bash
python -m venv venv
source venv/bin/activate # On Windows, use `venv\Scripts\activate`
```

## Usage <a name = "usage"></a>

To use scratchTorch, import the necessary modules and start building your neural network models. Here’s a basic example to demonstrate how to use the implemented layers:

```python
from Networks import NeuralNetwork
from Layers import FullyConnectedLayer
from ActivationFunctions import Tanh, ReLU
from LossFunctions import MeanSquareError
from Optimisers import MiniBatchGradientDescent, StochasticGradientDescent

net = NeuralNetwork()

net.add(FullyConnectedLayer(28 * 28, 100))  # input_shape=(1, 28*28)    ;   output_shape=(1100)
net.add(Tanh())
net.add(FullyConnectedLayer(100, 50))  # input_shape=(1, 100)      ;   output_shape=(1, 50)
net.add(Tanh())
net.add(FullyConnectedLayer(50, 10))  # input_shape=(1, 50)       ;   output_shape=(1, 10)
net.add(Tanh())

# train on 1000 samples
net.use_loss(MeanSquareError())
optimiser = StochasticGradientDescent(lr=0.1)
net.fit(x_train[0:1000], y_train[0:1000], optimiser, epochs=100)

```
