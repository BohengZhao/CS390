import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
import random

# Setting random seeds to keep everything deterministic.
random.seed(1618)
np.random.seed(1618)
# tf.set_random_seed(1618)   # Uncomment for TF1.
tf.random.set_seed(1618)

# Disable some troublesome logging.
# tf.logging.set_verbosity(tf.logging.ERROR)   # Uncomment for TF1.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Information on dataset.
NUM_CLASSES = 10
IMAGE_SIZE = 784

# Use these to set the algorithm to use.
# ALGORITHM = "guesser"


ALGORITHM = "custom_net"
# ALGORITHM = "tf_net"


class NeuralNetwork_2Layer():
    def __init__(self, inputSize, outputSize, neuronsPerLayer, learningRate=0.1, N=2, activationFunc="Sigmoid"):
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.neuronsPerLayer = neuronsPerLayer
        self.lr = learningRate
        self.W1 = np.random.randn(self.inputSize, self.neuronsPerLayer)
        self.W2 = np.random.randn(self.neuronsPerLayer, self.outputSize)
        self.layer = N
        self.activation = activationFunc

    # Activation function Sigmoid
    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # MSE loss function used in debugging
    def __mse(self, y, y_hat):
        return np.sum((y_hat - y) ** 2) / y.shape[1]

    # Activation prime function of Sigmoid
    def __sigmoidDerivative(self, x):
        return self.__sigmoid(x) * (1 - self.__sigmoid(x))

    # Activation function ReLU
    def __ReLU(self, x):
        return np.maximum(0, x)

    # Activation prime function of ReLU
    def __ReLUDerivative(self, x):
        x[x <= 0] = 0
        x[x > 0] = 1
        return x

    # Batch generator for mini-batches. Not randomized.
    def __batchGenerator(self, l, n):
        for i in range(0, len(l), n):
            yield l[i: i + n]

    def __learningRate(self, epoch, total):
        if epoch > 0.8 * total:
            return self.lr * 0.1
        elif epoch > 0.5 * total:
            return self.lr * 0.5
        else:
            return self.lr

    # Training with backpropagation.
    def train(self, xVals, yVals, epochs=10, minibatches=True, mbs=40):
        if minibatches is True:
            for epoch in range(0, epochs):
                batch_x = self.__batchGenerator(xVals, mbs)
                batch_y = self.__batchGenerator(yVals, mbs)
                for sample in range(0, xVals.shape[0], mbs):
                    small_batch_x = next(batch_x)
                    small_batch_y = next(batch_y)
                    L1out, L2out = self.__forward(small_batch_x)
                    L2e = L2out - small_batch_y
                    if self.activation == "Sigmoid":
                        L2d = L2e * self.__sigmoidDerivative(L2out)
                    elif self.activation == "ReLU":
                        L2d = L2e * self.__ReLUDerivative(np.dot(L1out, self.W2))
                    else:
                        raise Exception("NO Activation function specified")

                    L1e = np.dot(L2d, self.W2.T)

                    if self.activation == "Sigmoid":
                        L1d = L1e * self.__sigmoidDerivative(L1out)
                    elif self.activation == "ReLU":
                        L1d = L1e * self.__ReLUDerivative(np.dot(small_batch_x, self.W1))

                    L1a = small_batch_x.T.dot(L1d) * self.__learningRate(epoch, epochs)
                    L2a = L1out.T.dot(L2d) * self.__learningRate(epoch, epochs)
                    self.W1 -= L1a
                    self.W2 -= L2a
                print("Epoch: ", epoch + 1)
        else:
            for epoch in range(0, epochs):
                for sample in range(0, 1000):
                #for sample in range(0, xVals.shape[0]):
                    L1out, L2out = self.__forward(xVals[[sample], :])
                    L2e = L2out - yVals[[sample], :]
                    L2d = L2e * self.__sigmoidDerivative(L2out)
                    L1e = np.dot(L2d, self.W2.T)
                    L1d = L1e * self.__sigmoidDerivative(L1out)
                    L1a = xVals[[sample], :].T.dot(L1d) * self.lr
                    L2a = L1out.T.dot(L2d) * self.lr
                    self.W1 -= L1a
                    self.W2 -= L2a
            print("Epoch: ", epoch)

    # Forward pass.
    def __forward(self, input):
        if self.activation == "Sigmoid":
            layer1 = self.__sigmoid(np.dot(input, self.W1))
            layer2 = self.__sigmoid(np.dot(layer1, self.W2))
        elif self.activation == "ReLU":
            layer1 = self.__ReLU(np.dot(input, self.W1))
            layer2 = self.__ReLU(np.dot(layer1, self.W2))
        return layer1, layer2

    # Predict.
    def predict(self, xVals):
        _, layer2 = self.__forward(xVals)
        return layer2


# Classifier that just guesses the class label.
def guesserClassifier(xTest):
    ans = []
    for entry in xTest:
        pred = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        pred[random.randint(0, 9)] = 1
        ans.append(pred)
    return np.array(ans)


# =========================<Pipeline Functions>==================================

def getRawData():
    mnist = tf.keras.datasets.mnist
    (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
    print("Shape of xTrain dataset: %s." % str(xTrain.shape))
    print("Shape of yTrain dataset: %s." % str(yTrain.shape))
    print("Shape of xTest dataset: %s." % str(xTest.shape))
    print("Shape of yTest dataset: %s." % str(yTest.shape))
    return ((xTrain, yTrain), (xTest, yTest))


def preprocessData(raw):
    ((xTrain, yTrain), (xTest, yTest)) = raw  # TODO: Add range reduction here (0-255 ==> 0.0-1.0).
    xTrain = xTrain.reshape((xTrain.shape[0], xTrain.shape[1] * xTrain.shape[2]))
    xTest = xTest.reshape((xTest.shape[0], xTest.shape[1] * xTest.shape[2]))
    xTrain = xTrain / 255.0
    xTest = xTest / 255.0
    yTrainP = to_categorical(yTrain, NUM_CLASSES)
    yTestP = to_categorical(yTest, NUM_CLASSES)
    print("New shape of xTrain dataset: %s." % str(xTrain.shape))
    print("New shape of xTest dataset: %s." % str(xTest.shape))
    print("New shape of yTrain dataset: %s." % str(yTrainP.shape))
    print("New shape of yTest dataset: %s." % str(yTestP.shape))
    return ((xTrain, yTrainP), (xTest, yTestP))


def trainModel(data):
    xTrain, yTrain = data
    if ALGORITHM == "guesser":
        return None  # Guesser has no model, as it is just guessing.
    elif ALGORITHM == "custom_net":
        custom_nn = NeuralNetwork_2Layer(IMAGE_SIZE, 10, 512, activationFunc="ReLU")
        custom_nn.train(xTrain, yTrain, minibatches=True)
        print("Building and training Custom_NN.")
        print("Not yet implemented.")  # TODO: Write code to build and train your custom neural net.
        return custom_nn
    elif ALGORITHM == "tf_net":
        print("Building and training TF_NN.")
        print("Not yet implemented.")  # TODO: Write code to build and train your keras neural net.
        return None
    else:
        raise ValueError("Algorithm not recognized.")


def runModel(data, model):
    if ALGORITHM == "guesser":
        return guesserClassifier(data)
    elif ALGORITHM == "custom_net":
        print("Testing Custom_NN.")
        return model.predict(data)
    elif ALGORITHM == "tf_net":
        print("Testing TF_NN.")
        print("Not yet implemented.")  # TODO: Write code to run your keras neural net.
        return None
    else:
        raise ValueError("Algorithm not recognized.")


def evalResults(data, preds):  # TODO: Add F1 score confusion matrix here.
    xTest, yTest = data
    acc = 0
    for i in range(preds.shape[0]):
        if np.argmax(preds[i], 0) == np.argmax(yTest[i], 0):   acc = acc + 1
    accuracy = acc / preds.shape[0]
    print("Classifier algorithm: %s" % ALGORITHM)
    print("Classifier accuracy: %f%%" % (accuracy * 100))
    print()


# =========================<Main>================================================

def main():
    raw = getRawData()
    data = preprocessData(raw)
    model = trainModel(data[0])
    preds = runModel(data[1][0], model)
    evalResults(data[1], preds)


if __name__ == '__main__':
    main()
