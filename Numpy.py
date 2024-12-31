# 2 Layer Neural Network in NumPy

import numpy as np

# X = input of our 3 input XOR gate
# set up the inputs of the neural network (right from the table)
X = np.array([[0,0,0],[0,0,1],[0,1,0], \
              [0,1,1],[1,0,0],[1,0,1],[1,1,0], \
              [1,1,1]], dtype=float)

# y = our output of our neural network
y = np.array([[1], [0], [0], [0], \
              [0], [0], [1], [1]], dtype=float)

# what value we want to predict
xPredicted = np.array([[0,0,1]], dtype=float)

X = X/np.amax(X, axis=0) # maximum of X input array
# maximum of xPredicted (our input data for the prediction)
xPredicted = xPredicted/np.amax(xPredicted, axis=0)

# set up our Loss file for graphing
lossFile = open("SumSquaredLossList.csv", "w")

class Neural_Network(object):
    def __init__(self):
        # parameters
        self.inputLayerSize = 3    # X1,X2,X3
        self.outputLayerSize = 1   # Y1
        self.hiddenLayerSize = 4   # Size of the hidden layer

        # build weights of each layer
        # set to random values
        # look at the interconnection diagram to make sense of this
        # 3x4 matrix for input to hidden
        self.W1 = \
            np.random.rand(self.inputLayerSize, self.hiddenLayerSize)
        # 4x1 matrix for hidden layer to output
        self.W2 = \
            np.random.rand(self.hiddenLayerSize, self.outputLayerSize)
