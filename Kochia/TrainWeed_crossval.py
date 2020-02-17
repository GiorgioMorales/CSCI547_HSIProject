from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam, SGD
from keras.utils import to_categorical
from sklearn.model_selection import StratifiedKFold
import numpy as np
from operator import truediv
import h5py
from sklearn.decomposition import PCA
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, cohen_kappa_score, confusion_matrix

from networks import *

import keras.backend as k

k.set_image_data_format('channels_last')
k.set_learning_phase(1)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
LOAD HDF5 FILE
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

hdf5_file = h5py.File('weed_dataset_w25.hdf5', "r")
train_x = np.array(hdf5_file["train_img"][...])
# train_x = train_x / np.max(train_x)
# train_x = np.clip(train_x, 0, 1)
#train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], train_x.shape[2], train_x.shape[3], 1))
train_y = np.array(hdf5_file["train_labels"][...])

# Average consecutive bands
img2 = np.zeros((train_x.shape[0], train_x.shape[1], train_x.shape[2], int(train_x.shape[3]/2)))
for n in range(0, train_x.shape[0]):
    # Average consecutive bands
    for i in range(0, train_x.shape[3], 2):
        img2[n, :, :, int(i/2)] = (train_x[n, :, :, i] + train_x[n, :, :, i + 1]) / 2.

train_x = img2

print(train_x.shape)
train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], train_x.shape[2], train_x.shape[3], 1))

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
TRAIN PROPOSED NETWORK
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

windowSize = train_x.shape[1]
classes = 3
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
    
                
data = 'WEED'
# Load model
print("Loading model...")
model = hyper3dnet(img_shape=(windowSize, windowSize, train_x.shape[3], 1), classes=int(classes))
model.summary()

ntrain = 1
for train, test in kfold.split(train_x, train_y):
    k.set_learning_phase(1)

    ytrain = to_categorical(train_y[train]).astype(np.int32)
    ytest = to_categorical(train_y[test]).astype(np.int32)

    # Compile model
    model = hyper3dnet(img_shape=(windowSize, windowSize, train_x.shape[3], 1), classes=classes)
    model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['acc', categorical_accuracy])
    #optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    #model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc', categorical_accuracy])
    filepath = "weights-hyper3dnet" + data + str(ntrain) + "-best_3layers_4filters.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    ep = 100

    # Train model on dataset
    print(data + ": Training" + str(ntrain) + "begins...")
    history = model.fit(x=train_x[train], y=ytrain, validation_data=(train_x[test], ytest),
                        batch_size=32, epochs=ep, callbacks=callbacks_list)
                        
    # Evaluate network
    k.set_learning_phase(0)
    model.load_weights("weights-hyper3dnet" + data + str(ntrain) + "-best_3layers_4filters.h5")
    ypred = model.predict(train_x[test])
    
    # Calculate metrics
    oa = accuracy_score(np.argmax(ytest, axis=1), np.argmax(ypred, axis=-1))
    confusion = confusion_matrix(np.argmax(ytest, axis=1), np.argmax(ypred, axis=-1))
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(np.argmax(ytest, axis=1), np.argmax(ypred, axis=-1))
    prec, rec, f1, support = precision_recall_fscore_support(np.argmax(ytest, axis=1),
                                                            np.argmax(ypred, axis=-1), average='macro')
    
    # Add metrics to the list
    cvoa.append(oa * 100)
    cvaa.append(aa * 100)
    cvka.append(kappa * 100)
    cvpre.append(prec * 100)
    cvrec.append(rec * 100)
    cvf1.append(f1 * 100)
    
    print("%s: %.3f%%" % (model.metrics_names[1], oa * 100))
    file_name = "report_ntrain_" + str(ntrain) + ".txt"
    with open(file_name, 'w') as x_file:
       x_file.write("Overall accuracy%.3f%%" % (float(oa)))

    ntrain += 1

bestindex = np.argmax(cvoa) + 1
model.load_weights("weights-hyper3dnet" + data + str(bestindex) + "-best_3layers_4filters.h5")
model.save(data + "_hyper3dnet_4layers_8filters.h5")

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
