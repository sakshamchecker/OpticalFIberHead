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


def unet():
    EPOCHS = 35
    BATCH_SIZE = 32
    ImgHieght = 256
    ImgWidth = 256
    Channels = 3
    input_img = Input((ImgHieght, ImgWidth, 3), name='img')
    model = get_unet(input_img, n_filters=16, dropout=0.2, batchnorm=True)
    model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=[tf.keras.metrics.MeanSquaredError()])


    callbacks = [
        EarlyStopping(patience=10, verbose=1),
        ReduceLROnPlateau(factor=0.1, patience=5, min_lr=1e-5, verbose=1),
        ModelCheckpoint('model/model-unet.h5', verbose=1, save_best_only=True, save_weights_only=True)
    ]

    STEP_SIZE_TRAIN = timage_generator.n/BATCH_SIZE
    STEP_SIZE_VALID = vimage_generator.n/BATCH_SIZE



    results = model.fit(train_gen,
                        steps_per_epoch=STEP_SIZE_TRAIN,
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        callbacks=callbacks,
                        validation_data=valid_gen,
                        validation_steps=STEP_SIZE_VALID)



    from matplotlib import pyplot as plt
    import seaborn as sns
    plt.figure(figsize=(8, 8))
    plt.title("Learning curve")
    plt.plot(results.history["loss"], label="loss", color=sns.xkcd_rgb['greenish teal'])
    plt.plot(results.history["val_loss"], label="val_loss", color=sns.xkcd_rgb['amber'])
    plt.plot( np.argmin(results.history["val_loss"]), np.min(results.history["val_loss"]), marker="x", color="r", label="best model")
    plt.xlabel("Epochs")
    plt.ylabel("log_loss")
    plt.legend()
    # plt.grid(False)
    plt.show()


def linknet_t():
    EPOCHS = 35
    BATCH_SIZE = 32
    ImgHieght = 256
    ImgWidth = 256
    Channels = 3
    model=lnm
    model.compile(optimizer='adam', loss="binary_crossentropy", metrics=[tf.keras.metrics.MeanSquaredError()])
    callbacks = [
        EarlyStopping(patience=10, verbose=1),
        ReduceLROnPlateau(factor=0.1, patience=5, min_lr=1e-5, verbose=1),
        ModelCheckpoint('model/model-linknet.h5', verbose=1, save_best_only=True, save_weights_only=True)
    ]
    model.summary()
    STEP_SIZE_TRAIN = timage_generator.n/BATCH_SIZE
    STEP_SIZE_VALID = vimage_generator.n/BATCH_SIZE
    results = model.fit(train_gen,
                        steps_per_epoch=STEP_SIZE_TRAIN,
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        callbacks=callbacks,
                        validation_data=valid_gen,
                        validation_steps=STEP_SIZE_VALID)
    from matplotlib import pyplot as plt
    import seaborn as sns
    plt.figure(figsize=(8, 8))
    plt.title("Learning curve")
    plt.plot(results.history["loss"], label="loss", color=sns.xkcd_rgb['greenish teal'])
    plt.plot(results.history["val_loss"], label="val_loss", color=sns.xkcd_rgb['amber'])
    plt.plot( np.argmin(results.history["val_loss"]), np.min(results.history["val_loss"]), marker="x", color="r", label="best model")
    plt.xlabel("Epochs")
    plt.ylabel("log_loss")
    plt.legend()
    # plt.grid(False)
    plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--query', type=str, default='', help='model to train')
    args = parser.parse_args()
    
    if args.query == 'unet':
        unet()
    if args.query=='linknet':
        linknet_t()