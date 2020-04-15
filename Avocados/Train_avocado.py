from keras.callbacks import ModelCheckpoint
# from keras.optimizers import Adam
from sklearn.model_selection import StratifiedKFold
import numpy as np
from operator import truediv
import h5py
import cv2
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, cohen_kappa_score, confusion_matrix
import pickle
from scipy.signal import find_peaks
from Avocados.networks import *

import keras.backend as k

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Data augmentation functions
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


def add_rotation_flip(x, y):
    x = np.reshape(x, (x.shape[0], x.shape[1], x.shape[2], x.shape[3], 1))

    # Flip horizontally
    x_h = np.flip(x[:, :, :, :, :], 1)
    # Flip vertically
    x_v = np.flip(x[:, :, :, :, :], 2)
    # Flip horizontally and vertically
    x_hv = np.flip(x_h[:, :, :, :, :], 2)

    # Concatenate
    x = np.concatenate((x, x_h, x_v, x_hv))
    y = np.concatenate((y, y, y, y))

    return x, y


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
LOAD HDF5 FILE
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

hdf5_file = h5py.File('avocado_dataset_w64.hdf5', "r")
train_x = np.array(hdf5_file["train_img"][...])

train_y = np.array(hdf5_file["train_labels"][...])

# Average consecutive bands
img2 = np.zeros((train_x.shape[0], int(train_x.shape[1] / 2), int(train_x.shape[2] / 2), int(train_x.shape[3] / 2)))
for n in range(0, train_x.shape[0]):
    xt = cv2.resize(np.float32(train_x[n, :, :, :]), (32, 32), interpolation=cv2.INTER_CUBIC)
    for i in range(0, train_x.shape[3], 2):
        img2[n, :, :, int(i / 2)] = (xt[:, :, i] + xt[:, :, i + 1]) / 2.

train_x = img2

# Select most relevant bands
nbands = 5

count = 0
with open('avocadoselection.p', 'rb') as f:
    saliency = pickle.load(f)

peaks, _ = find_peaks(saliency, height=5, distance=5)

saliency = np.flip(np.argsort(saliency))

indexes = []
for i in range(0, len(saliency)):
    if saliency[i] in peaks:
        indexes.append(saliency[i])

indexes = indexes[0:nbands]

indexes.sort()

temp = np.zeros((train_x.shape[0], train_x.shape[1], train_x.shape[2], nbands))

for nb in range(0, nbands):
    temp[:, :, :, nb] = train_x[:, :, :, indexes[nb]]

train_x = temp

train_x, train_y = add_rotation_flip(train_x, train_y)
print(train_x.shape)
# train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], train_x.shape[2], train_x.shape[3]))

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
TRAIN PROPOSED NETWORK
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

windowSize = train_x.shape[1]
classes = 2
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
cvoa = []
cvaa = []
cvka = []
cvpre = []
cvrec = []
cvf1 = []
cva1 = []
cva2 = []
cva3 = []


def categorical_accuracy(y_true, y_pred):
    return k.cast(k.equal(k.argmax(y_true, axis=-1),
                          k.argmax(y_pred, axis=-1)),
                  k.floatx())


def AA_andEachClassAccuracy(confusion_m):
    list_diag = np.diag(confusion_m)
    list_raw_sum = np.sum(confusion_m, axis=1)
    each_ac = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_ac)
    return each_ac, average_acc


data = 'AVOCADO'
# Load model
print("Loading model...")
model = hyper3dnet2(img_shape=(windowSize, windowSize, train_x.shape[3], 1), classes=int(classes))
model.summary()

ntrain = 1
for train, test in kfold.split(train_x, train_y):
    ytrain = train_y[train]
    ytest = train_y[test]
    xtrain = train_x[train]
    xtest = train_x[test]

    # Compile model
    model = hyper3dnet2(img_shape=(windowSize, windowSize, train_x.shape[3], 1), classes=classes)
    model.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['acc'])
    # optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    # model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['acc'])
    filepath = "weights5-hyper3dnet" + data + str(ntrain) + "-best_3layers_4filters.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    ep = 600

    # Train model on dataset
    print(data + ": Training" + str(ntrain) + "begins...")
    history = model.fit(x=xtrain, y=ytrain, validation_data=(xtest, ytest),
                        batch_size=16, epochs=ep, callbacks=callbacks_list)

    # Evaluate network
    model.load_weights("weights5-hyper3dnet" + data + str(ntrain) + "-best_3layers_4filters.h5")
    model.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['acc'])
    ypred = model.predict(xtest)
    ypred = ypred.round()

    # Calculate metrics
    oa = accuracy_score(ytest, ypred)
    confusion = confusion_matrix(ytest, ypred)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(ytest, ypred)
    prec, rec, f1, support = precision_recall_fscore_support(ytest, ypred, average='macro')

    # Add metrics to the list
    cvoa.append(oa * 100)
    cvaa.append(aa * 100)
    cvka.append(kappa * 100)
    cvpre.append(prec * 100)
    cvrec.append(rec * 100)
    cvf1.append(f1 * 100)

    print("%s: %.3f%%" % (model.metrics_names[1], oa * 100))
    file_name = "report_ntrain_selected" + str(ntrain) + ".txt"
    with open(file_name, 'w') as x_file:
        x_file.write("Overall accuracy%.3f%%" % (float(oa)))

    ntrain += 1

bestindex = np.argmax(cvoa) + 1
model.load_weights("weights5-hyper3dnet" + data + str(bestindex) + "-best_3layers_4filters.h5")
model.save(data + "_hyper3dnet_4layers_8filters_selected5.h5")

file_name = "classification_report_" + data + ".txt"
with open(file_name, 'w') as x_file:
    x_file.write("Overall accuracy%.3f%% (+/- %.3f%%)" % (float(np.mean(cvoa)), float(np.std(cvoa))))
    x_file.write('\n')
    x_file.write("Average accuracy%.3f%% (+/- %.3f%%)" % (float(np.mean(cvaa)), float(np.std(cvaa))))
    x_file.write('\n')
    x_file.write("Kappa accuracy%.3f%% (+/- %.3f%%)" % (float(np.mean(cvka)), float(np.std(cvka))))
    x_file.write('\n')
    x_file.write("Precision accuracy%.3f%% (+/- %.3f%%)" % (float(np.mean(cvpre)), float(np.std(cvpre))))
    x_file.write('\n')
    x_file.write("Recall accuracy%.3f%% (+/- %.3f%%)" % (float(np.mean(cvrec)), float(np.std(cvrec))))
    x_file.write('\n')
    x_file.write("F1 accuracy%.3f%% (+/- %.3f%%)" % (float(np.mean(cvf1)), float(np.std(cvf1))))
