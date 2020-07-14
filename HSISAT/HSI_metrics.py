import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from scipy import stats

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Plot confusion matrix
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

with open('metrics/IP/meanshyper3dnetIP', 'rb') as f:
    means = pickle.load(f)
with open('metrics/IP/stdshyper3dnetIP', 'rb') as f:
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
classes_list = list(range(0, int(3)))
plt.figure()
plot_confusion_matrix(means, stds, classescf=classes_list)
dataset = 'IP'
# plt.savefig('MatrixConfusion_' + dataset + 'hyper3dnet.png', dpi=1200)


# Box-plot
with open('metrics/IP/cvf1hyper3dnetIP', 'rb') as f:
    cvf1 = pickle.load(f)
with open('metrics/IP/cvf1hyper3dnetsigmoidIP', 'rb') as f:
    cvf1a = pickle.load(f)

df = pd.DataFrame({'Hyper3DNet': cvf1, 'Hyper3DNet-Attention': cvf1a})
df[['Hyper3DNet', 'Hyper3DNet-Attention']].plot(kind='box')
print(stats.ttest_rel(df['Hyper3DNet'], df['Hyper3DNet-Attention']))
# plt.savefig('Box_' + dataset + '6bands.png', dpi=1200)
