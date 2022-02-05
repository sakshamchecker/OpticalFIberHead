import numpy as np 
import pandas as pd
import os
from itertools import chain
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from sklearn.model_selection import train_test_split

import tensorflow as tf
from skimage.color import rgb2gray
from tensorflow.keras import Input
from tensorflow.keras.models import Model, load_model, save_model
from tensorflow.keras.layers import Input, Activation, BatchNormalization, Dropout, Lambda, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


from utilities.data_gen import *
from utilities.unet import get_unet
from utilities.linknet import model as lnm
from matplotlib import pyplot as plt
import seaborn as sns





def unet(input_im):
    EPOCHS = 35
    BATCH_SIZE = 32
    ImgHieght = 256
    ImgWidth = 256
    Channels = 3

    input_img = Input((ImgHieght, ImgWidth, 3), name='img')
    model = get_unet(input_img, n_filters=16, dropout=0.2, batchnorm=True)
    # model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=[tf.keras.metrics.MeanSquaredError()])

    
    model.load_weights('model/model-unet.h5')


    import cv2
    im = cv2.imread(input_im)

    img = cv2.resize(im ,(ImgHieght, ImgWidth))
    img = img / 255
    img = img[np.newaxis, :, :, :]

    pred=model.predict(img)


    plt.figure(figsize=(12,12))
    plt.subplot(1,2,1)
    plt.imshow(np.squeeze(img))
    plt.title('Original Image')
    plt.subplot(1,2,2)
    plt.imshow(np.squeeze(pred))
    plt.title('Prediction')
    plt.show()






    

def linknet_t(input_im):
    EPOCHS = 35
    BATCH_SIZE = 32
    ImgHieght = 256
    ImgWidth = 256
    Channels = 3
    model=lnm
    model.load_weights('model/model-linknet.h5')
    import cv2
    im = cv2.imread(input_im)

    img = cv2.resize(im ,(ImgHieght, ImgWidth))
    img = img / 255
    img = img[np.newaxis, :, :, :]

    pred=model.predict(img)


    plt.figure(figsize=(12,12))
    plt.subplot(1,2,1)
    plt.imshow(np.squeeze(img))
    plt.title('Original Image')
    plt.subplot(1,2,2)
    plt.imshow(np.squeeze(pred))
    plt.title('Prediction')
    plt.show()



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--query', type=str, help='model to use')
    parser.add_argument('--image', type=str, help='image to test')

    args = parser.parse_args()
    
    if args.query == 'unet':
        unet(args.image)
    if args.query=='linknet':
        linknet_t(args.image)