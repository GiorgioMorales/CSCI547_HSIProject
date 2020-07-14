from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
from sklearn.model_selection import StratifiedKFold

from HSISAT.networks import *
from HSISAT.read_data import *

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
classes = np.max(y)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Training
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Load model
print("Loading model...")
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
    data = 'IP'
    model = hyper3dnet(img_shape=(windowSize, windowSize, X.shape[3], 1), classes=int(classes) + 1)
    filesave = "weights/IP/weights-hyper3dnet" + dataset + str(ntrain) + "-best.h5"
    model = transfermodel(modelt=model, src=filesave)
    model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['acc'])

    # checkpoint
    filepath = "weights-hyper3dnetsigmoid" + dataset + str(ntrain) + "-best.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    if dataset == 'IP':
        ep = 10
    else:
        ep = 40

    # Train model on dataset
    print(dataset + ": Training" + str(ntrain) + "begins...")
    history = model.fit(x=X[train], y=ytrain, validation_data=(X[test], ytest),
                        batch_size=8, epochs=ep, callbacks=callbacks_list)

    ntrain += 1
