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
from tensorflow_model_optimization.sparsity import keras as sparsity

from HSISAT.networks import *
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
xtrain, xtest, ytrain, ytest = splitraintestset(X, y, 0.3)
xtrain = np.reshape(xtrain, (xtrain.shape[0], xtrain.shape[1], xtrain.shape[2], xtrain.shape[3], 1))
xtest = np.reshape(xtest, (xtest.shape[0], xtest.shape[1], xtest.shape[2], xtest.shape[3], 1))
classes = np.max(y)


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Training
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
cvoa = []
cvaa = []
cvka = []
cvpre = []
cvrec = []
cvf1 = []

# Load model
print("Loading model...")
model = hyper3dnet(img_shape=(windowSize, windowSize, X.shape[3], 1), classes=int(classes) + 1)
model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['acc'])
model.summary()

ntrain = 1
for train, test in kfold.split(X, y):

    ytrain = to_categorical(y[train]).astype(np.int32)
    ytest = to_categorical(y[test]).astype(np.int32)

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    PRUNING
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    batch_size = 8

    epochs = 4
    end_step = np.ceil(1.0 * len(ytrain) / batch_size).astype(np.int32) * epochs
    print(end_step)

    new_pruning_params = {
        'pruning_schedule': sparsity.PolynomialDecay(initial_sparsity=0.50,
                                                      final_sparsity=0.90,
                                                      begin_step=0,
                                                      end_step=end_step,
                                                      frequency=100)
    }

    model.load_weights()

    new_pruned_model = sparsity.prune_low_magnitude(loaded_model, **new_pruning_params)
    new_pruned_model.summary()

    # # Compile model
    # model = attention1(img_shape=(windowSize, windowSize, X.shape[3], 1), attention=False, classes=int(classes) + 1)
    # optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    # model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'])

    # # checkpoint
    # filepath = "weights-attention" + data + str(ntrain) + "-attention-best.h5"
    # checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    # callbacks_list = [checkpoint]

    # if data == 'IP':
    #     ep = 50
    # else:
    #     ep = 40


# checkpoint
filepath = "weights-auto-best.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

history = model.fit(x=xtrain, y=ytrain, validation_data=(xtest, ytest), epochs=300, batch_size=8)
