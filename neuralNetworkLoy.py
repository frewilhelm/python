# This neural network script is based on the remarks of Aidan Wilson
#  https://towardsdatascience.com/inroduction-to-neural-networks-in-python-7e0b422e6c24

# A neural network contains an input (x), some hidden and an output layer (y). Between each
#  layer, there is a set of weights and biases. Finally, there is an activation function
#  for each hidden layer.


# Import libraries
import numpy as np
import matplotlib.pyplot as plt

# Input data (x)
inputData = np.array([[0, 1, 0],
                      [0, 1, 1],
                      [0, 1, 1],
                      [1, 0, 0],
                      [1, 1, 1],
                      [1, 0, 1]])

# Output data (y) (for training)
outputData = np.array([[0],
                       [0],
                       [0],
                       [1],
                       [1],
                       [1]])

# 2-Layer neural network
class neuralNetwork:

    # Init-Functions
    def __init__(self, inputData, outputData):
        self.input = inputData                          # Input layer
        self.outputData = outputData                    # Output layer

        self.weights = np.array([[.70], [.80], [.50]])  # Weights

        # For Graph
        self.error_history = []
        self.epoch_list = []

    # Activation-Function: Sigmoid (logistic funtion)
    # {0,1}, nonlinear
    def activationFunc(self, x, derivation=False):
        if derivation==True:
            return x * (1 - x)
        return 1 / (1 + np.exp(-x))

    # Feedforward: Calculate predicted output y
    def feedForward(self):
        self.hidden = self.activationFunc(np.dot(self.input, self.weights))

    # Backpropagation: Update weights and biases
    def backpropagation(self):
        self.error = self.outputData - self.hidden
        delta = self.error * self.activationFunc(self.hidden, derivation=True)
        self.weights += np.dot(self.input.T, delta)

    # Training
    # epochs - Times of weight-update
    def train(self, epochs = 25000):
        for epoch in range(epochs):
            self.feedForward()
            self.backpropagation()

            self.error_history.append(np.average(np.abs(self.error)))
            self.epoch_list.append(epoch)

    # predict
    def predict(self, newInput):
        predictions = self.activationFunc(np.dot(newInput, self.weights))
        return predictions


NN = neuralNetwork(inputData, outputData) # Create Object

NN.train()

example = np.array([[1, 1, 0]])
example_2 = np.array([[0, 1, 1]])

print(NN.predict(example), ' - Correct: ', example[0][0])
print(NN.predict(example_2), ' - Correct: ', example_2[0][0])

plt.figure(figsize=(15,5))
plt.plot(NN.epoch_list, NN.error_history)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
