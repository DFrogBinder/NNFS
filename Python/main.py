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
class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities 
class Loss:
    # calculate the data and the regularization loss,
    # given model output and ground truth values
    def calculate(self, output,y):
        # Calculate the sample loss
        sample_losses = self.forward(output,y)

        # Calculate the mean loss
        data_loss = np.mean(sample_losses)

        return data_loss

# Cross-entropy loss computation
class Loss_CategoricalCrossEntropy(Loss):
    # Forward Pass
    def forward(self,y_pred,y_true):
        #Get number of samples in a batch 
        samples = len(y_pred)

        # Clip data to prevent diviion by zero
        y_pred_clipped = np.clip(y_pred,1e-7, 1- 1e-7)
        # Probablities for target values
        # only if categorial labels
        if len(y_true.shape) == 1:
            correct_confidence = y_pred_clipped[
                range(samples),
                y_true]
        # Mask values for one-hot encoded labels
        elif len(y_true.shape) == 2:
            correct_confidence = np.sum(
                y_pred_clipped * y_true, axis=1
            )
        # Losses
        negative_log_likelihood = -np.log(correct_confidence)
        return negative_log_likelihood


dense1 = Layer_Dense(2,3) # Initializes a Layer_Dense object with 4 inputs and 5 neurons (This calls the __init__ method)
dense2 = Layer_Dense(3,3)

activation1 = Activation_ReLU()
activation2 = Activation_Softmax()

loss_function = Loss_CategoricalCrossEntropy()

dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

loss = loss_function.calculate(activation2.output,y)