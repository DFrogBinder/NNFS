import numpy as np

np.random.seed(0)

X = [[1, 2, 3, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]


class Layer_Dense:
    # Network initialization function
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons) # Initializes initial weigths for the model
        self.biases = np.zeros((1, n_neurons)) # Initilizes initial bias matrix (as 0s)

    # Forward method (Propagates inputs through the network to the next layer)
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

layer1 = Layer_Dense(4,5) # Initializes a Layer_Dense object with 4 inputs and 5 neurons (This calls the __init__ method)
layer2 = Layer_Dense(5,2)

layer1.forward(X) # Pass data through Layer_Dens object
# print(layer1.output)
layer2.forward(layer1.output)
print(layer2.output)
