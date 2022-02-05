
# !pip install git+https://github.com/qubvel/segmentation_models
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

import segmentation_models

from segmentation_models import Linknet
import os
# os.environ['SM_FRAMEWORK'] = 'tf.keras'
import segmentation_models as sm

print('sm.version=' + sm.__version__)
sm.set_framework('tf.keras')



model=Linknet()