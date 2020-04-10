import warnings

warnings.filterwarnings("ignore")

import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture
import pickle
from hierarchical_clustering import hierarchical_clustering as hc
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib
from scipy import integrate
from selector import Selector
import utils

waves = utils.imager_waves


class Transformer:
    '''
    Class to fit a given dataset using kernel density
    estimation, then transform the data using the
    resulting Gaussian
    '''

    def __init__(self, produce, method, histogram, bandwidth, num_waves=5,
                 cluster_reduce=False, visualize=False):
        self.produce = produce
        self.method = method
        self.histogram = histogram
        self.bandwidth = bandwidth
        self.num_waves = num_waves
        self.cluster_reduce = cluster_reduce
        self.visualize = visualize
        self.waves = utils.imager_waves

    def _load_data(self):
        '''
        Load the data in. If cluster reduce is true,
        use hierarchical clustering to keep only the
        data in the cluster with the highest average
        score
        '''

        # Path for the raw population numbers
        path_name = 'results/selected_wavelengths/'
        file_name = self.produce + '_' + str(self.num_waves) + '_' + self.method + '.txt'
        path = path_name + file_name
        data = open(path, 'r')

        all_wavelengths = []
        scores = []

        # Read in the data
        for line in data:
            split_line = line.split(',')
            wave = split_line[0].replace('[', '').replace(']', '')
            all_wavelengths.append(wave)
            scores.append(split_line[1].strip())

        scores = np.array(scores)

        # Sort each of the arrays of selected wavelengths
        temp_waves = []
        for w in all_wavelengths:
            temp_waves.append(np.array(sorted([int(item) for item in w.split()])))
        all_wavelengths = temp_waves

        if self.visualize:
            in_nanometers = np.array([np.take(self.waves, w) for w in all_wavelengths])
            plt.hist(np.array(in_nanometers), bins=290)
            plt.ylim((0, 150))
            plt.tick_params(labelsize=36)
            plt.title('GA Wavelength Distribution before Clustering', fontsize=44)
            plt.xlabel('Wavelength (nm)', fontsize=39)
            plt.ylabel('Frequency', fontsize=39)
            labels = ['First Wavelength', 'Second Wavelength', 'Third Wavelength',
                      'Fourth Wavelength', 'Fifth Wavelength']
            plt.legend(labels, fontsize=30)
            plt.show()

        # Use hierarchical clustering to select the cluster with
        # the highest average score
        if self.cluster_reduce:
            all_wavelengths = self.wave_selector._cluster_reduce(all_wavelengths, scores)

            if self.visualize:
                in_nanometers = np.array([np.take(self.waves, w) for w in all_wavelengths])
                plt.hist(np.array(in_nanometers), bins=290)
                plt.ylim((0, 150))
                plt.tick_params(labelsize=36)
                plt.title('GA Wavelength Distribution after Clustering', fontsize=44)
                plt.xlabel('Wavelength (nm)', fontsize=39)
                plt.ylabel('Frequency', fontsize=39)
                labels = ['First Wavelength', 'Second Wavelength', 'Third Wavelength',
                          'Fourth Wavelength', 'Fifth Wavelength']
                plt.legend(labels, fontsize=30)
                plt.show()

        return np.array(all_wavelengths)

    def _generic_gaussians(self, indices, bandwidth):
        '''
        Given the indices of the wavelength centers and the
        filter bandwidth (given in nanometers), return the
        gaussians defined by these inputs
        '''

        average_step = (np.max(self.waves) - np.min(self.waves)) / float(len(self.waves))
        index_bandwidth = bandwidth / average_step  # Convert the bandwidth in nm to indices
        all_gaussians = []

        # Given the bandwidth (which is equivalent to the full width-
        # half maximum of a Gaussian), calculate the standard deviation
        # FWHM = 2*sqrt(2ln(2))*stdev => stdev = FWHM/(2*sqrt(2ln(2)))
        stdev = np.divide(index_bandwidth, np.multiply(2, np.sqrt(np.multiply(np.log(2), 2))))

        for ind in indices:
            curve = np.linspace(0, 290, 290)
            pdf = mlab.normpdf(curve, ind, stdev)
            # pdf = np.multiply(pdf, np.divide(highest_count, np.max(pdf)))
            gauss = np.divide(pdf, np.max(pdf))  # Normalize to 1
            all_gaussians.append(gauss)

        return all_gaussians

    def _get_rgb_gaussians(self):
        '''
        Return Gaussians associated with the RGB profile from
        a Nikon camera
        '''

        means = [113, 69, 33]
        sigmas = [6.25120179413, 8.35986952579, 5.35080953855]
        amplitudes = [0.455190125047, 0.394008187803, 0.961964168877]
        all_gaussians = []

        for i, mean in enumerate(means):
            curve = np.linspace(0, 290, 290)
            pdf = mlab.normpdf(curve, mean, sigmas[i])
            gauss = np.divide(pdf, np.max(pdf))  # Normalize to 1
            gauss = np.multiply(gauss, amplitudes[i])
            all_gaussians.append(gauss)

        return all_gaussians

    def _get_rgbnir_gaussians(self):
        '''
        Return known Gaussians for RGB+NIR filters
        '''

        means = [113, 69, 33, 224]
        sigmas = [6.25120179413, 8.35986952579, 5.35080953855, 6.25120179413]
        amplitudes = [0.455190125047, 0.394008187803, 0.961964168877, 0.455190125047]
        all_gaussians = []

        for i, mean in enumerate(means):
            curve = np.linspace(0, 290, 290)
            pdf = mlab.normpdf(curve, mean, sigmas[i])
            gauss = np.divide(pdf, np.max(pdf))  # Normalize to 1
            gauss = np.multiply(gauss, amplitudes[i])
            all_gaussians.append(gauss)

        return all_gaussians

    def transform_data(self):
        '''
        Get the original produce data, then transofrm that data
        using the histogram used to fit the histogram of wavelengths
        '''

        if self.bandwidth and (self.method == 'ga'):
            self.wave_selector = Selector(self.produce, self.method, self.num_waves, self.histogram)
            gaussians = self._extract_gaussians()
        else:
            self.wave_selector = Selector(self.produce, self.method, self.num_waves, self.histogram)
            indices = self.wave_selector._select_wavelengths()
            here = True
            if self.method == 'rgb':
                gaussians = self._get_rgb_gaussians()
            elif self.method == 'rgbnir':
                gaussians = self._get_rgbnir_gaussians()
            else:
                bandwidth = 20  # Measured in nanometers
                gaussians = self._generic_gaussians(indices, bandwidth)

        prefix = 'data/for_classification/'
        path_name = prefix + self.produce + '.p'

        # Load in the dataset
        try:
            produce_spectras = pickle.load(open(path_name, 'rb'))

        except (OSError, IOError) as e:
            print ('Error locating specified pickle file')
            return None

        new_data = []
        for spectrum in produce_spectras:
            old_spectrum = spectrum[0]
            new_point = []

            # For each gaussian, integrate under the curve
            # multiplied by the original data point to
            # simulate a filter with the bandwidth of the
            # gaussian
            for g in gaussians:
                curve = np.multiply(old_spectrum, g)
                integral = integrate.trapz(curve)
                new_point.append(integral)

            # new_spectrum = np.multiply(old_spectrum, gaussian)

            new_datum = (new_point, spectrum[1], spectrum[2])
            new_data.append(new_datum)

        return new_data

    def _extract_gaussians(self):
        '''
        Recover the gaussian distribution parameters
        (for self.num_waves gaussians). It is assumed
        that the number of gaussians is equal to the
        number of selected wavelengths, and that
        hierarchical clustering has parsed out the
        single gaussian that contributes to the highest
        average score in the case the distributions of
        the entire population are multimodal.
        '''

        all_wavelengths = self._load_data()
        gaussian_mixture = np.zeros(290)
        all_gaussians = []

        # Loop over each column of the selected wavelengths
        # and fit a gaussian distribution to each, saving
        # the estimated parameters
        for i in range(self.num_waves):
            gm = GaussianMixture(n_components=1, max_iter=200)
            selected_waves = all_wavelengths[:, i].reshape(-1, 1)
            highest_count = max([np.count_nonzero(selected_waves == elem) for elem in selected_waves])
            num_bins = np.max(selected_waves) - np.min(selected_waves)

            gm.fit(selected_waves)
            mean = gm.means_[0][0]
            print (int(round(mean)))
            spread = gm.covariances_[0][0][0]

            stdev = np.sqrt(spread)
            curve = np.linspace(0, 290, 290)
            pdf = mlab.normpdf(curve, mean, stdev)
            pdf = np.multiply(pdf, np.divide(highest_count, np.max(pdf)))
            gauss = np.divide(pdf, np.max(pdf))  # Normalize to 1
            all_gaussians.append(gauss)

            gaussian_mixture = np.add(gaussian_mixture, pdf)

            if self.visualize:
                plt.plot(pdf)

        if self.visualize:
            plt.title('GA Population Gaussian Fits - ' + self.produce + ' - ' + str(self.num_waves))
            plt.hist(all_wavelengths.flatten(), bins=290, color='green')
            plt.plot(gaussian_mixture, color='red')
            plt.show()

        # Normalize the gaussian to 1.0 and return
        # return np.divide(gaussian_mixture, np.max(gaussian_mixture))

        return all_gaussians
