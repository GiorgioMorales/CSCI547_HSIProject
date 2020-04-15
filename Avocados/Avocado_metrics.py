import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
import numpy as np
from operator import truediv
import h5py
import tensorflow as tf
import cv2
from scipy.signal import find_peaks
from scipy import stats
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, cohen_kappa_score, confusion_matrix

from Avocados.networks import *

import keras.backend as k
import pickle
import pandas as pd

k.set_image_data_format('channels_last')
k.set_learning_phase(0)

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

indesex = [0, 74, 75, 135, 140]

indexes.sort()
print(indexes)

temp = np.zeros((train_x.shape[0], train_x.shape[1], train_x.shape[2], nbands))

for nb in range(0, nbands):
    temp[:, :, :, nb] = train_x[:, :, :, indexes[nb]]

# temp = np.zeros((train_x.shape[0], train_x.shape[1], train_x.shape[2], 5))
#
# temp[:, :, :, 0] = train_x[:, :, :,33]
# temp[:, :, :, 1] = train_x[:, :, :,74]
# temp[:, :, :, 2] = train_x[:, :, :,97]
# temp[:, :, :, 3] = train_x[:, :, :,121]
# temp[:, :, :, 4] = train_x[:, :, :,145]


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


dataset = 'AVOCADO'
data = 'AVOCADO'
# Load model
print("Loading model...")
model = hyper3dnet2(img_shape=(windowSize, windowSize, train_x.shape[3], 1), classes=int(classes))
model.summary()

confmatrices = np.zeros((10, int(classes), int(classes)))

ntrain = 1
for train, test in kfold.split(train_x, train_y):
    ytrain = train_y[train]
    ytest = train_y[test]

    # Data augmentation
    xtrain = train_x[train]
    xtest = train_x[test]

    # Compile model
    model = hyper3dnet2(img_shape=(windowSize, windowSize, train_x.shape[3], 1), classes=classes)
    model.load_weights("weights5PLS-hyper3dnet" + data + str(ntrain) + "-best_3layers_4filters.h5")
    model.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['acc'])
    ypred = model.predict(xtest)
    ypred = ypred.round()

    sess = tf.Session()
    with sess.as_default():
        con_mat = tf.math.confusion_matrix(labels=ytest,
                                           predictions=ypred).eval()

    con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=3)
    classes_list = list(range(0, int(classes)))
    con_mat_df = pd.DataFrame(con_mat_norm, index=classes_list, columns=classes_list)

    confmatrices[ntrain - 1, :, :] = con_mat_df.values

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

    ntrain += 1

file_name = "classification_report_hyper3dnet_5bands_" + dataset + "_PLS.txt"
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

# Calculate mean and std
means = np.mean(confmatrices * 100, axis=0)
stds = np.std(confmatrices * 100, axis=0)

with open('meanshyper3dnet5PLS', 'wb') as f:
    pickle.dump(means, f)
with open('stdshyper3dnet5PLS', 'wb') as f:
    pickle.dump(stds, f)
with open('cvf1hyper3dnet5PLS', 'wb') as f:
    pickle.dump(cvf1, f)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Plot confusion matrix
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

with open('band_selection\\5bands\\meanshyper3dnet5', 'rb') as f:
    means = pickle.load(f)
with open('band_selection\\5bands\\stdshyper3dnet5', 'rb') as f:
    stds = pickle.load(f)


def plot_confusion_matrix(cm, cms, classescf,
                          cmap=plt.cm.Blues):
    """
   This function prints and plots the confusion matrix.
   Normalization can be applied by setting `normalize=True`.
   """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    tick_marks = np.arange(len(classescf))
    plt.xticks(tick_marks, classescf, rotation=45)
    plt.yticks(tick_marks, classescf)

    thresh = cm.max() / 2.

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):

            if (cm[i, j] == 100 or cm[i, j] == 0) and cms[i, j] == 0:
                plt.text(j, i, '{0:.0f}'.format(cm[i, j]) + '\n$\pm$' + '{0:.0f}'.format(cms[i, j]),
                         horizontalalignment="center",
                         verticalalignment="center", fontsize=20,
                         color="white" if cm[i, j] > thresh else "black")

            else:
                plt.text(j, i, '{0:.2f}'.format(cm[i, j]) + '\n$\pm$' + '{0:.2f}'.format(cms[i, j]),
                         horizontalalignment="center",
                         verticalalignment="center", fontsize=20,
                         color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# Plot non-normalized confusion matrix
classes_list = list(range(0, int(2)))
plt.figure()
plot_confusion_matrix(means, stds, classescf=classes_list)
dataset = 'AVOCADO'
plt.savefig('MatrixConfusion_' + dataset + '_pruned_5bands.png', dpi=1200)

# Box-plot
with open('t-test/cvf1hyper3dnet5', 'rb') as f:
    cvf1 = pickle.load(f)
with open('t-test/cvf1hyper3dnet5_NC_OC_IE', 'rb') as f:
    cvf2 = pickle.load(f)
with open('t-test/cvf1hyper3dnet5GA', 'rb') as f:
    cvf3 = pickle.load(f)
with open('t-test/cvf1hyper3dnet5PLS', 'rb') as f:
    cvf4 = pickle.load(f)

df = pd.DataFrame({'SSA': cvf1, 'OCF': cvf2, 'HAGRID': cvf3,
                   'PLS-DA': cvf4})
df[['SSA', 'OCF', 'HAGRID', 'PLS-DA']].plot(kind='box')
print(stats.ttest_rel(df['SSA'], df['OCF']))
print(stats.ttest_rel(df['SSA'], df['HAGRID']))
print(stats.ttest_rel(df['SSA'], df['PLS-DA']))
plt.savefig('Box_' + dataset + '.png', dpi=1200)
