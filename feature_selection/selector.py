# Simple selection of wavelengths without binning or applying Gaussian filters

import pickle
import numpy as np
from sklearn.mixture import GaussianMixture
from hierarchical_clustering import hierarchical_clustering as hc


class Selector:

    def __init__(self, produce, method, num_waves, histogram):
        self.produce = produce
        self.method = method
        self.num_waves = num_waves
        self.histogram = histogram

    def transform_data(self):
        """
        Identify the subset of wavelengths to use, then transform
        the original data to consist of only those wavelengths
        identified
        """

        indices = self._select_wavelengths()
        original_data = self._load_data()
        new_data = [(np.take(d[0], indices), d[1], d[2]) for d in original_data]

        return new_data

    def _load_data(self):
        prefix = 'data/for_classification/'
        path_name = prefix + self.produce + '.p'

        # Load in the dataset
        try:
            produce_spectras = pickle.load(open(path_name, 'rb'))
            return produce_spectras

        except (OSError, IOError) as e:
            print ('Error locating specified pickle file')
            return None

    def _select_wavelengths(self):
        """
        Retrieve the subset of wavelengths determined by the specified method
        """

        if self.method == 'pls':
            indices = self._get_pls_indices()

        elif self.method == 'ga':
            indices = self._get_ga_indices()
            ga_out = open('results/selected_wavelengths/' + 'avocado' + '_' + str(len(indices)) + '_ga_selected.txt', 'w')
            ga_out.write(str(indices))
            ga_out.close()
        elif self.method == 'rgb':  # Wavelengths associated with red, green and blue light
            indices = np.array([33, 69, 113])

        elif self.method == 'rgbnir':  # Wavelengths associated with RGB light and an additional near infra-red channel
            indices = np.array([33, 69, 113, 224])

        elif self.method == 'all':
            indices = np.arange(146)

        return indices

    def _get_ga_indices(self):
        """
        Return the indices selected by the GA. Either use the standard "best"
        member of the population if self.histogram is False, or use the
        histogram-based analysis if self.histogram is True
        """

        wavelengths, scores = self._get_ga_data()

        if not self.histogram:
            best_ind = np.argmax(scores)
            best_waves = wavelengths[best_ind]

        elif self.histogram:
            best_waves = self._gaussian_peak_centers(wavelengths, scores)
            print (best_waves)

        return np.sort(best_waves)

    def _gaussian_peak_centers(self, wavelengths, scores):
        """
        Fit the Gaussian mixture model to the histogram of the population
        after hierarchical clustering has been performed and the best
        cluster selected
        """

        population = self._cluster_reduce(wavelengths, scores)

        gaussian_mixture = np.zeros(146)
        means = []

        # Loop over each column of the selected wavelengths
        # and fit a gaussian distribution to each, saving
        # the estimated parameters
        for i in range(self.num_waves):
            gm = GaussianMixture(n_components=1, max_iter=200)
            selected_waves = population[:, i].reshape(-1, 1)
            highest_count = max([np.count_nonzero(selected_waves == elem) for elem in selected_waves])
            num_bins = np.max(selected_waves) - np.min(selected_waves)

            gm.fit(selected_waves)
            mean = gm.means_[0][0]
            means.append(mean)

        indices = [int(round(m)) for m in means]
        return indices

    def _cluster_reduce(self, wavelengths, scores):
        """
        Use hierarchical clustering to cluster the population, then
        return the cluster with the highest average score
        """

        best_score = 0.0  # Score of the best (in terms of average score) cluster
        best_cluster = None
        keeping = 0

        biggest = None  # The largest cluster found
        biggest_size = 0
        biggest_score = 0  # Score of the biggest cluster

        clust = hc(wavelengths, self.produce, self.num_waves)
        clusters = clust.cluster()

        # Find the average score of each cluster
        for clust_number in set(clusters):
            score = 0.0
            in_cluster = 0.0
            for i, cluster in enumerate(clusters):
                if cluster == clust_number:
                    score += float(scores[i])
                    in_cluster += 1.0

            score = score / in_cluster

            # Keep track of the best cluster found so far (as long as at least 100 members)
            if score > best_score and in_cluster >= 100:
                best_score = score
                best_cluster = clust_number
                keeping = in_cluster

            # Also keep track of the largest cluster found
            if in_cluster > biggest_size:
                biggest = clust_number
                biggest_size = in_cluster
                biggest_score = score

        if keeping == 0:  # No cluster had 100 members, then take the biggest cluster
            best_cluster = biggest
            print ('Number of points in cluster: ', biggest_size)
            print ('Average score for the cluster: ', biggest_score)

        else:
            print ('Number of points in cluster: ', keeping)
            print ('Average score for the cluster: ', best_score)
        temp_wavelengths = []
        for i, cluster in enumerate(clusters):
            if cluster == best_cluster:
                temp_wavelengths.append(wavelengths[i])

        subpopulation = np.array(temp_wavelengths)

        return subpopulation

    def _get_ga_data(self):
        """
        Read in the GA data from file
        """

        path_name = 'results/selected_wavelengths/'
        file_name = self.produce + '_' + str(self.num_waves) + '_' + self.method + '.txt'
        path = path_name + file_name
        data = open(path, 'r')

        wavelengths = []
        scores = []
        for line in data:
            print line
            split_line = line.split(',')
            wave = split_line[0].replace('[', '').replace(']', '').split()
            wave = sorted([int(w) for w in wave])
            wavelengths.append(wave)
            scores.append(split_line[1].strip())

        scores = np.array(scores)

        return wavelengths, scores

    def _get_pls_indices(self):
        """
        Return the subset of wavelength indices selected by
        the PLS-DA wavelength selection method.
        """

        path_name = 'results/selected_wavelengths/'
        file_name = self.produce + '_' + str(self.num_waves) + '_' + self.method + '.txt'
        path = path_name + file_name
        data = open(path, 'r')

        for line in data:
            last_line = line

        indices = last_line.strip('[').strip(']').split()
        indices = [int(i) for i in indices]
        return (np.array(indices))

#
