import numpy as np
from keras import backend
from sklearn.model_selection import StratifiedKFold

from mlp import MLP
from selector import Selector
from transformer import Transformer
import h5py

# Set the parameters for what dataset to use
produce = 'avocado'
num_wavelengths = 20  # Number of wavelengths to select, -1 to use all wavelengths
# There must be a file with the name <produce>_<num_wavelengths>_ga.txt
# in the 'results/selected_wavelengths' folder
method = 'ga'  # Can be 'all', 'rgb', 'rgbnir', 'ga', or 'pls'
histogram = True  # If True, return wavelengths associated with the peaks of
# the histogram of the population (this is the HAGRID method)
# If False, return the population best (standard GA)
# Used only with method = 'ga'
transform = False  # Transform the data to mimic a multispectral imager (i.e. specify
# bandwith in addition to wavelengths selected)
bandwidth = False  # Use histogram of the population to determine the bandwidth of the filter
# Used only in conjunction with transform = True and method = 'ga'
# If false, use 20 nm standard bandwith to transform data
visualize = False  # Displays some plots
reduce_cluster = True  # If True, use hierarchical clustering to cluster the population and
# select the cluster whose members have the highest average fitness
# Should usually be set to True

if transform:
    data_transformer = Transformer(produce, method, histogram, bandwidth, num_wavelengths, reduce_cluster, visualize)
    new_data = data_transformer.transform_data()

else:
    wave_selector = Selector(produce, method, num_wavelengths, histogram)
    new_data = wave_selector.transform_data()


# x = [p[0] for p in new_data]
# # Encode fresh as 0, discount as 1, and old as 2
# y = [0 if p[1] == 'fresh' else 1 if p[1] == 'old' else 2 for p in new_data]

hdf5_file = h5py.File('C:\\Users\\w63x712\\Documents\\Machine_Learning\\'
                      'CSCI547_HSIProject\\Avocados\\avocado_dataset_w64.hdf5', "r")
x = np.array(hdf5_file["train_img"][...])
y = np.array(hdf5_file["train_labels"][...])

# Average consecutive bands
img2 = np.zeros((x.shape[0], x.shape[1], x.shape[2], int(x.shape[3] / 2)))
for n in range(0, x.shape[0]):
    # Average consecutive bands
    for i in range(0, x.shape[3], 2):
        img2[n, :, :, int(i / 2)] = (x[n, :, :, i] + x[n, :, :, i + 1]) / 2.

x = img2
x = np.mean(x, axis=1)
x = np.mean(x, axis=1)

s = np.arange(x.shape[0])
np.random.shuffle(s)

train_x = x[s, :]
train_y = y[s]
train_x = train_x[0:15, :]
train_y = train_y[0:15]

new_data = []
count = 0
for sp in x:
    sublist = [sp, y[count]]
    new_data.append(sublist)
    count += 1

# MLP parameters for the grid search
lrs = [0.001]
hidden_sizes = [3]

# Variables for keeping track of the best solution found
final_score = 0
final_stdev = 0
final_scores = []
final_lr = None
final_nodes = None

for lr in lrs:
    for hidden in hidden_sizes:

        hist = ''
        if histogram:
            hist = 'histogram'
            if bandwidth:
                hist = 'histogram - bandwidth'
        print (produce + ' - ' + method + ' - ' + str(num_wavelengths) + ' - ' + hist + ':', lr, hidden)
        # Stratified validation
        skf = StratifiedKFold(n_splits=10, shuffle=True)
        skf.get_n_splits(x, y)

        # List to track cross-validation scores for summary statistics
        scores = []

        # Initialize the confusion matrix
        conf_mat = None

        # Perform cross-validation
        for train_ind, test_ind in skf.split(x, y):
            train = np.take(new_data, train_ind, axis=0)
            test = np.take(new_data, test_ind, axis=0)

            # Feed-forward neural network
            model = MLP(train)
            model.train()
            score = round(model.score(test), 4)
            scores.append(score)
            conf_mat = model.confusion_matrix(test, conf_mat)
            backend.clear_session()

        if np.average(scores) > final_score:
            final_score = np.average(scores)
            final_scores = scores
            final_stdev = np.std(scores)
            final_lr = lr
            final_nodes = hidden

        # Display summary statistics
        print ('	------------------------')
        print ('	All scores: ', scores)
        print ('	Mean: ', np.average(scores))
        print ('	Min: ', min(scores))
        print ('	Max: ', max(scores))
        print ('	StDev:', np.std(scores))
        print ('	------------------------')
        # print ('	',conf_mat)
        print ('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')

hist = ''
if histogram:
    hist = 'histogram'
    if bandwidth:
        hist = 'histogram - bandwidth'
print (
    'Best for ' + produce + ' - ' + method + ' - ' + str(num_wavelengths) + ' - ' + hist + ':', final_lr, final_nodes)
print ('	Mean: ', final_score)
print ('	Stdev: ', final_stdev)
print ('	All: ', final_scores)
