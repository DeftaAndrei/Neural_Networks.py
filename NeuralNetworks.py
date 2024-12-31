import numpy as np

class Neural_Network:
    def activationSigmoidPrime(self, s):
        # First derivative of sigmoid activation function
        return s * (1 - s)

    def saveSumSquaredLossLists(self, i, error):
        with open("SumSquaredLossList.csv", "a") as lossFile:
            lossFile.write(str(i) + "," + str(error.tolist()) + '\n')

    def saveWeights(self):
        np.savetxt("weightsLayer1.txt", self.W1, fmt="%s")
        np.savetxt("weightsLayer2.txt", self.W2, fmt="%s")

    def predictOutput(self, xPredicted):
        print("Predicted XOR output data based on trained weights: ")
        print("Expected (X1-X3): \n" + str(xPredicted))
        print("Output (Y1): \n" + str(self.feedForward(xPredicted)))

# Example usage:
# Instantiate the neural network
myNeuralNetwork = Neural_Network()

# Define training parameters
trainingEpochs = 1000

for i in range(trainingEpochs):
    print("Epoch: " + str(i) + "\n")
    print("Network Input: \n" + str(X))
    print("Expected Output of XOR Gate Neural Network: \n" + str(y))

    # Feedforward
    actualOutput = myNeuralNetwork.feedForward(X)
    print("Actual Output from XOR Gate Neural Network: \n" + str(actualOutput))

    # Calculate loss
    loss = np.mean(np.square(y - actualOutput))
    myNeuralNetwork.saveSumSquaredLossLists(i, loss)
    print("Mean Sum Squared Loss: " + str(loss))
    print("\n")

    # Backpropagation
    myNeuralNetwork.backPropagate(X, y, actualOutput)

# Save weights and predict final output
myNeuralNetwork.saveWeights()
myNeuralNetwork.predictOutput(xPredicted)
