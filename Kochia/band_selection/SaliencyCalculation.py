import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import find_peaks
from sklearn import mixture
import itertools
from scipy.stats import norm

# Parameters
components = 20
flag_smooth = False  # Tag used to smooth the curve using a 1-D convolution


def select(saliency, nbands):
    """Method to calculate the peaks of a vector (not used for GMM fitting)"""
    peaks, _ = find_peaks(saliency, distance=4)

    saliency = np.flip(np.argsort(saliency))

    indexes = []
    for ind in range(0, len(saliency)):
        if saliency[ind] in peaks:
            indexes.append(saliency[ind])

    print(indexes)
    indexes = indexes[0:nbands]

    return indexes


###############################################
#       Read the Saliency Vectors
###############################################

SA = np.zeros((10, 150,))

for i in range(0, 10):
    with open('SA_vectors/SA_fold_pruning' + str(i + 1), 'rb') as f:  # Comparison
        # -Pruning_vs_NoPruning
        SA[i][:] = pickle.load(f)

# Calculate the mean and standard deviation
means = np.mean(SA, axis=0)
stds = np.std(SA, axis=0)

###############################################
#       Calculate Saliency Metric
###############################################

sal = means  # Alternative saliency metric

# Print the selected bands
select1 = select(means, components)
select1.sort()
print("Selected bands using the mean as the metric: " + str(select1))
select2 = select(sal, components)
select2.sort()
print("Selected bands using the mean/std as the metric: " + str(select2))

# Plot means and stds
fig, ax = plt.subplots()
clrs = sns.color_palette()
with sns.axes_style("darkgrid"):
    epochs = list(range(150))

    ax.plot(epochs, means, c=clrs[0])
    ax.fill_between(epochs, means - stds, means + stds, alpha=0.3, facecolor=clrs[0])
    ax.legend()
    ax.set_yscale('log')

###############################################
#                GMM fitting
###############################################

color_iter = itertools.cycle(['navy', 'c', 'cornflowerblue', 'gold', 'darkorange'])

plt.figure()
plt.plot(sal)

if flag_smooth:
    box = np.ones(3) / 3
    sal = np.convolve(sal, box, mode='same')  # savgol_filter(means, 5, 3)  #
else:
    sal = sal

select2 = select(sal, components)
select2.sort()

# Plot Saliency vector before and after smoothing
plt.plot(sal, color='red')

# Transform the curve into a histogram
hist = []
for i in range(len(sal)):
    for r in range(int(sal[i])):
        hist.append(i)

hist = np.array(hist)
X = hist.reshape(-1, 1)

# Fit a Gaussian mixture with EM
gmm = mixture.GaussianMixture(n_components=components, covariance_type='full',
                              max_iter=100).fit(X)

means_hat = gmm.means_.flatten()
weights_hat = gmm.weights_.flatten()
sds_hat = np.sqrt(gmm.covariances_).flatten()

print(gmm.converged_)
print(means_hat)
print(sds_hat)
print(weights_hat)

plt.figure()
plt.hist(X, bins=150, density=True, alpha=0.6, color='g')
for g in range(components):
    mu1_h, sd1_h = means_hat[g], sds_hat[g]
    x_axis = np.linspace(mu1_h - 10 * sd1_h, mu1_h + 10 * sd1_h, 150)
    plt.plot(x_axis, norm.pdf(x_axis, mu1_h, sd1_h) * weights_hat[g])

means_hat.sort()
print("Selected bands using the mean/std as the metric: " + str(np.floor(means_hat)))

# import random
#
# random.seed(7)  # to make it reproducible
#
# # create a mixture of normals:
# #  1000 from N(0, 1)
# #  2000 from N(6, 2)
# mix = np.concatenate([np.random.normal(2, 1, [1000]),
#                       np.random.normal(7, 2, [2000]),
#                       np.random.normal(20, 3, [600]),
#                       np.random.normal(28, 3, [600])])
#
# # histogram:
# plt.figure()
# plt.hist(mix, bins=120)
#
#
# plt.figure()
# plt.hist(mix, bins=150, density=True, alpha=0.6, color='g')
# X = mix.reshape(-1, 1)

# # Fit a Gaussian mixture with EM
# gmm = mixture.GaussianMixture(n_components=2, covariance_type='full',
#                               max_iter=100).fit(X)
# means_hat = gmm.means_.flatten()
# weights_hat = gmm.weights_.flatten()
# sds_hat = np.sqrt(gmm.covariances_).flatten()
# for g in range(2):
#     mu1_h, sd1_h = means_hat[g], sds_hat[g]
#     x_axis = np.linspace(mu1_h - 10 * sd1_h, mu1_h + 10 * sd1_h, 120)
#     plt.plot(x_axis, norm.pdf(x_axis, mu1_h, sd1_h) * weights_hat[g])
