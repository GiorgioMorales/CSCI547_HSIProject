import os
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import load_model
from sklearn.model_selection import StratifiedKFold
import numpy as np
from operator import truediv
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, cohen_kappa_score, confusion_matrix
from random import shuffle

from Autoencoder.read_data import *
from Autoencoder.autoencoder import *

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
DATA LOADER
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# GLOBAL VARIABLES
dataset = 'IP'  # ['IP', 'PU', 'SA']
windowSize = 25

X, y = loadata(dataset)
X, y = createImageCubes(X, y, window=windowSize)
xtrain, xtest, ytrain, ytest = splitraintestset(X, y, 0.3)
xtrain = np.reshape(xtrain, (xtrain.shape[0], xtrain.shape[1], xtrain.shape[2], xtrain.shape[3], 1))
xtest = np.reshape(xtest, (xtest.shape[0], xtest.shape[1], xtest.shape[2], xtest.shape[3], 1))
classes = np.max(y)


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Training
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Load model
# print("Loading model...")
# model = autoencoder2d(img_shape=(windowSize, windowSize, X.shape[3], 1))
# model.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['acc', 'mse'])
# model.summary()
#
# # checkpoint
# filepath = "weights-auto-best.h5"
# checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
# callbacks_list = [checkpoint]
#
# history = model.fit(x=xtrain, y=xtrain, validation_data=(xtest, xtest), epochs=30, batch_size=8)
