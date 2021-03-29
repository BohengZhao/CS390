import math
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import copy
import tensorflow.keras.backend as K
import random
import imageio
from skimage.transform import resize
from scipy.optimize import fmin_l_bfgs_b   # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_l_bfgs_b.html
from tensorflow.keras.applications import vgg19
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import warnings

random.seed(1618)
np.random.seed(1618)
#tf.set_random_seed(1618)   # Uncomment for TF1.
tf.random.set_seed(1618)

#tf.logging.set_verbosity(tf.logging.ERROR)   # Uncomment for TF1.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

CONTENT_IMG_PATH = "./pics/monster2.jpg"
STYLE_IMG_PATH = "./pics/style1.jpg"
tf.compat.v1.disable_eager_execution()

CONTENT_IMG_H = 500
CONTENT_IMG_W = 500

STYLE_IMG_H = 500
STYLE_IMG_W = 500

CONTENT_WEIGHT = 0.0001    # Alpha weight.
STYLE_WEIGHT = 0.9999      # Beta weight.
STYLE_LAYER_WEIGHT = [0.2, 0.2, 0.2, 0.2, 0.2]
TOTAL_WEIGHT = 0.0005

TRANSFER_ROUNDS = 5

#=============================<Helper Fuctions>=================================
'''
TODO: implement this.
This function should take the tensor and re-convert it to an image.
'''
def deprocessImage(img):

    x = img.reshape((CONTENT_IMG_H, CONTENT_IMG_W, 3))
    # Remove zero-center by mean pixel
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype("uint8")
    return x


def gramMatrix(x):
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram

#========================<Loss Function Builder Functions>======================

def styleLoss(style, gen):
    G = gramMatrix(gen)
    A = gramMatrix(style)
    return K.sum(K.square(G - A)) / (4.0 * math.pow(style.shape[1], 2) * math.pow((STYLE_IMG_H * STYLE_IMG_W), 2))

def contentLoss(content, gen):
    return K.sum(K.square(gen - content))


def totalVariationLoss(x): # designed to keep the generated image locally coherent
    a = K.square(x[:, : CONTENT_IMG_H - 1, : CONTENT_IMG_W - 1, :] - x[:, 1:, : CONTENT_IMG_W - 1, :])
    b = K.square(x[:, : CONTENT_IMG_H - 1, : CONTENT_IMG_W - 1, :] - x[:, : CONTENT_IMG_W - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25)) #https://keras.io/examples/generative/neural_style_transfer/


#=========================<Pipeline Functions>==================================

def getRawData():
    print("   Loading images.")
    print("      Content image URL:  \"%s\"." % CONTENT_IMG_PATH)
    print("      Style image URL:    \"%s\"." % STYLE_IMG_PATH)
    cImg = load_img(CONTENT_IMG_PATH)
    tImg = cImg.copy()
    sImg = load_img(STYLE_IMG_PATH)
    print("      Images have been loaded.")
    return ((cImg, CONTENT_IMG_H, CONTENT_IMG_W), (sImg, STYLE_IMG_H, STYLE_IMG_W), (tImg, CONTENT_IMG_H, CONTENT_IMG_W))



def preprocessData(raw):
    img, ih, iw = raw
    img = img_to_array(img)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        img = resize(img, (ih, iw, 3))
    img = img.astype("float64")

    #imgsave = deprocessImage(img)
    #saveFile = "Initial.png"
    #imageio.imwrite(saveFile, img)

    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img


'''
TODO: Allot of stuff needs to be implemented in this function.
First, make sure the model is set up properly.
Then construct the loss function (from content and style loss).
Gradient functions will also need to be created, or you can use K.Gradients().
Finally, do the style transfer with gradient descent.
Save the newly generated and deprocessed images.
'''
def styleTransfer(cData, sData, tData):
    print("   Building transfer model.")
    contentTensor = K.variable(cData, dtype=tf.float64)
    styleTensor = K.variable(sData, dtype=tf.float64)
    genPlaceholder = K.placeholder((CONTENT_IMG_H * CONTENT_IMG_W * 3), dtype=tf.float64)
    genTensor = K.reshape(genPlaceholder, (1, CONTENT_IMG_H, CONTENT_IMG_W, 3))

    inputTensor = K.concatenate([contentTensor, styleTensor, genTensor], axis=0)
    model = vgg19.VGG19(include_top=False, weights="imagenet", input_tensor=inputTensor)
    outputDict = dict([(layer.name, layer.output) for layer in model.layers])
    print("   VGG19 model loaded.")
    loss = tf.zeros(shape=())

    styleLayerNames = ["block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1", "block5_conv1"]
    contentLayerName = "block5_conv2"

    print("   Calculating content loss.")
    contentLayer = outputDict[contentLayerName]
    contentOutput = contentLayer[0, :, :, :]
    genOutput = contentLayer[2, :, :, :]
    loss += contentLoss(contentOutput, genOutput) * CONTENT_WEIGHT

    print("   Calculating style loss.")
    layer_index = 0
    for layerName in styleLayerNames:
        styleLayer = outputDict[layerName]
        styleOutput = styleLayer[1, :, :, :]
        genOutput = styleLayer[2, :, :, :]
        loss += styleLoss(styleOutput, genOutput) * STYLE_WEIGHT * STYLE_LAYER_WEIGHT[layer_index]
        layer_index += 1

    loss += totalVariationLoss(tf.cast(genTensor, tf.float32)) * TOTAL_WEIGHT
    outputs = [loss]

    grads = K.gradients(loss, genPlaceholder)
    outputs.append(grads)
    kFunction = K.function([genPlaceholder], outputs)

    x = tData.flatten()

    print("   Beginning transfer.")
    for i in range(TRANSFER_ROUNDS):
        print("   Step %d." % i)
        x, tLoss, info = fmin_l_bfgs_b(func=kFunction, x0=x, maxiter=30)
        print("      Loss: %f." % tLoss)
    img = deprocessImage(x)
    saveFile = "Transfer.png"
    imageio.imwrite(saveFile, img)

    os.system("say Image transfered")
    print("      Image saved to \"%s\"." % saveFile)
    print("   Transfer complete.")

#=========================<Main>================================================

def main():
    print("Starting style transfer program.")
    raw = getRawData()
    cData = preprocessData(raw[0])   # Content image.
    sData = preprocessData(raw[1])   # Style image.
    tData = preprocessData(raw[2])   # Transfer image.
    styleTransfer(cData, sData, tData)
    print("Done. Goodbye.")



if __name__ == "__main__":
    main()
