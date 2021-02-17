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


#ALGORITHM = "custom_net"
ALGORITHM = "tf_net"


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
        return np.maximum(x, 0)

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
    def train(self, xVals, yVals, epochs=1, minibatches=True, mbs=40):
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
                        L2d = L2e * self.__ReLUDerivative(L2out)
                    else:
                        raise Exception("NO Activation function specified")

                    L1e = np.dot(L2d, self.W2.T)

                    if self.activation == "Sigmoid":
                        L1d = L1e * self.__sigmoidDerivative(L1out)
                    elif self.activation == "ReLU":
                        L1d = L1e * self.__ReLUDerivative(L1out)
                    x = small_batch_x.T.dot(L1d)
                    L1a = small_batch_x.T.dot(L1d) * self.__learningRate(epoch, epochs)
                    L2a = L1out.T.dot(L2d) * self.__learningRate(epoch, epochs)
                    self.W1 -= L1a
                    self.W2 -= L2a
                print("Epoch: ", epoch + 1)
        else:
            for epoch in range(0, epochs):
                for sample in range(0, xVals.shape[0]):
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
    ((xTrain, yTrain), (xTest, yTest)) = raw
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
        print("Building and training Custom_NN.")
        custom_nn = NeuralNetwork_2Layer(IMAGE_SIZE, 10, 512, activationFunc="ReLU")
        custom_nn.train(xTrain, yTrain, minibatches=True)
        return custom_nn
    elif ALGORITHM == "tf_net":
        print("Building and training TF_NN.")
        model = keras.Sequential()
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dropout(0.2, input_shape=(IMAGE_SIZE,)))
        model.add(tf.keras.layers.Dense(512, activation=tf.nn.relu))
        model.add(tf.keras.layers.Dropout(0.2, input_shape=(512,)))
        model.add(tf.keras.layers.Dense(256, activation=tf.nn.relu))
        model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
        model.add(tf.keras.layers.Dropout(0.5, input_shape=(128,)))
        model.add(tf.keras.layers.Dense(64, activation=tf.nn.relu))
        model.add(tf.keras.layers.Dense(32, activation=tf.nn.relu))
        model.add(tf.keras.layers.Dropout(0.5, input_shape=(32,)))
        model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
        lossType = keras.losses.categorical_crossentropy
        model.compile(optimizer='adam', loss=lossType, metrics=['accuracy'])
        model.fit(xTrain, yTrain, epochs=5)
        return model
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
        return model.predict(data)
    else:
        raise ValueError("Algorithm not recognized.")


def evalResults(data, preds):  # TODO: Add F1 score confusion matrix here.
    xTest, yTest = data

    confusionMatrix = np.zeros((yTest.shape[1] + 1, yTest.shape[1] + 1)) # square matrix, +1 for total column and row
    acc = 0

    for i in range(preds.shape[0]):
        if np.argmax(preds[i], 0) == np.argmax(yTest[i], 0):
            acc = acc + 1
            confusionMatrix[np.argmax(preds[i], 0)][np.argmax(yTest[i], 0)] += 1
        else:
            confusionMatrix[np.argmax(preds[i], 0)][np.argmax(yTest[i], 0)] += 1
    accuracy = acc / preds.shape[0]

    sumColumn = confusionMatrix.sum(axis=0)
    sumRow = confusionMatrix.sum(axis=1)
    for i in range(0, yTest.shape[1] + 1):
        confusionMatrix[yTest.shape[1]][i] = sumColumn[i]
        confusionMatrix[i][yTest.shape[1]] = sumRow[i]
    confusionMatrix[yTest.shape[1]][yTest.shape[1]] = confusionMatrix.sum(axis=0)[yTest.shape[1]]

    print("CONFUSION MATRIX")
    space = "       "
    print("      ", "0", space, "1", space,"2", space,"3", space,"4", space,"5", space,"6", space,"7", space,"8", space,"9", space, "TOTAL")
    space = "     "
    for i in range(confusionMatrix.shape[0]):
        if i != confusionMatrix.shape[0] - 1:
            print(i, space, end="")
        else:
            print("TOTAL  ", end="")
        for j in range(confusionMatrix.shape[0]):
            print(confusionMatrix[i][j], end=(10-len(confusionMatrix[i][j].__str__()))*" ")
        print()
    F1_score = []
    for i in range(confusionMatrix.shape[0] - 1):
        precision = confusionMatrix[i][i] / confusionMatrix[i][-1]
        recall = confusionMatrix[i][i] / confusionMatrix[-1][i]
        F1_score.append((2 * ((precision * recall) / (precision + recall))).__str__())
    print()
    print("F1-Score:")
    print("      ", end="")
    for i in range(len(F1_score)):
        print("%.7s    " % F1_score[i], end="")
    print()



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
