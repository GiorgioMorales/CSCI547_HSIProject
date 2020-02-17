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

from networks import *
from Autoencoder.read_data import *

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
X = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2], X.shape[3], 1))
# xtrain, xtest, ytrain, ytest = splitraintestset(X, y, 0.3)
# xtrain = np.reshape(xtrain, (xtrain.shape[0], xtrain.shape[1], xtrain.shape[2], xtrain.shape[3], 1))
# xtest = np.reshape(xtest, (xtest.shape[0], xtest.shape[1], xtest.shape[2], xtest.shape[3], 1))
classes = np.max(y)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Training
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Load model
print("Loading model...")
# model = hyper3dnet(img_shape=(windowSize, windowSize, X.shape[3], 1), classes=int(classes) + 1)
# model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['acc'])
# model.summary()

# # checkpoint
# filepath = "weights-auto-best.h5"
# checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
# callbacks_list = [checkpoint]
#
# history = model.fit(x=xtrain, y=ytrain, validation_data=(xtest, ytest), epochs=300, batch_size=8)

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
cvoa = []
cvaa = []
cvka = []
cvpre = []
cvrec = []
cvf1 = []

ntrain = 1
for train, test in kfold.split(X, y):

    ytrain = to_categorical(y[train]).astype(np.int32)
    ytest = to_categorical(y[test]).astype(np.int32)

    # Compile model
    model = hyper3dnet(img_shape=(windowSize, windowSize, X.shape[3], 1), classes=int(classes) + 1)
    model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['acc'])

    # checkpoint
    filepath = "weights-hyper3dnet" + dataset + str(ntrain) + "-best.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    if dataset == 'IP':
        ep = 50
    else:
        ep = 40

    # Train model on dataset
    print(dataset + ": Training" + str(ntrain) + "begins...")
    history = model.fit(x=X[train], y=ytrain, validation_data=(X[test], ytest),
                        batch_size=8, epochs=ep, callbacks=callbacks_list)
