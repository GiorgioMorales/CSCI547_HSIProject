####################################################
#                                                  #
# Main file for setting the parameters and running #
# the various wavelength selection algorithms. The #
# run_ga_selector runs the custom genetic          #
# algorithm delevoped for this research.           #
#                                                  #
# The algorithms include the genetic algorithm,    #
# partial least squares, and successive            #
# projections algorithm.                           #
#                                                  #
####################################################

# Bad practice, but the warnings were getting annoying
import warnings
# Other imports
from wavelength_selection import PLSDASelector, GASelector, SPASelector
import numpy as np
import matplotlib.pyplot as plt
import utils
import h5py
from random import shuffle

warnings.filterwarnings("ignore")

imager_waves = utils.imager_waves


def run_ga_selector(produce, wavelengths=0, cr=0.5, mr=0.1, tourny_size=5):
    """
    Run the genetic algorithm based wavelength selection algorithm. If
    wavelengths is set to zero, the GA will attempt to find the optimal number
    of wavelengths to use.

    Params:
        produce: string - type of produce to analyze. The selected produce type
                          should be located in the directory pointed to in
                          utils.load_spectra()
        wavelengths: int - number of wavelengths to be selected by the algorithm
        cr: float - crossover rate used by the GA
        mr: float - mutation rate used by the GA
        tourny_size: int - tournament size used by GA in tournament selection

    Returns:
        wave - array of selected wavelenths (in nanometers) of best individual
        ind - array of the indices corresponding to the selected wavelengths
        (population, scores) - tuple of every member of the GA population and
                               the corresponding scores achieved by the
                               classification algorithm used by the GA on each
                               member of the population
    """

    # produce_spectras = utils.load_spectra(produce)
    # data = [spectra[0:2] for spectra in produce_spectras]

    classifier = 'dtree'  # Base classifier for use in fitness function of GA
    select = GASelector(data, wavelengths, classifier, mutation_rate=mr,
                        crossover_rate=cr, tourny_size=tourny_size)

    wave, ind = select.find_wavelengths()
    return wave, ind, (select.gen_alg.population, select.gen_alg.scores)


def run_pls_selector(data, produce, wavelengths):
    """
    Run the partial least squares based
    wavelength selection algorithm

    Params:
        produce: string - type of produce to analyze. The selected produce type
                          should be located in the directory pointed to in
                          utils.load_spectra()
        wavelengths: int - number of wavelengths to be selected by the algorithm

    Returns:
        wave - array of selected wavelenths (in nanometers)
        ind - array of the indices corresponding to the selected wavelengths
    """

    # produce_spectras = utils.load_spectra(produce)
    # data = [spectra[0:2] for spectra in produce_spectras]
    select = PLSDASelector(data, wavelengths)

    wave, ind = select.find_wavelengths()
    return wave, ind


def run_spa_selector(produce, wavelengths):
    """
    Run the successive projections based
    wavelength selection algorithm

    Params:
        produce: string - type of produce to analyze. The selected produce type
                          should be located in the directory pointed to in
                          utils.load_spectra()
        wavelengths: int - number of wavelengths to be selected by the algorithm

    Returns:
        wave - array of selected wavelenths (in nanometers)
        ind - array of the indices corresponding to the selected wavelengths
    """

    produce_spectras = utils.load_spectra(produce)
    data = [spectra[0:2] for spectra in produce_spectras]
    select = SPASelector(data, wavelengths)

    wave, ind = select.find_wavelengths()
    return wave, ind


if __name__ == '__main__':

    produce = 'kochia'
    num_waves = 20  # Number of wavelengths to select for subset
    write = True  # Write the selected wavelengths to an output file

    # Read data

    if produce == 'kochia':
        hdf5_file = h5py.File('C:\\Users\\w63x712\\Documents\\Machine_Learning\\'
                              'CSCI547_HSIProject\\Kochia\\weed_dataset_w25.hdf5', "r")
    else:
        hdf5_file = h5py.File('C:\\Users\\w63x712\\Documents\\Machine_Learning\\'
                              'CSCI547_HSIProject\\Avocados\\avocado_dataset_w64.hdf5', "r")

    train_x = np.array(hdf5_file["train_img"][...])
    train_y = np.array(hdf5_file["train_labels"][...])

    # Average consecutive bands
    img2 = np.zeros((train_x.shape[0], train_x.shape[1], train_x.shape[2], int(train_x.shape[3] / 2)))
    for n in range(0, train_x.shape[0]):
        # Average consecutive bands
        for i in range(0, train_x.shape[3], 2):
            img2[n, :, :, int(i / 2)] = (train_x[n, :, :, i] + train_x[n, :, :, i + 1]) / 2.

    train_x = img2

    train_x = np.mean(train_x, axis=1)
    train_x = np.mean(train_x, axis=1)

    s = np.arange(train_x.shape[0])
    np.random.shuffle(s)

    train_x = train_x[s, :]
    train_y = train_y[s]
    # train_x = train_x[0:1500, :]
    # train_y = train_y[0:1500]

    data = []
    count = 0
    for sp in train_x:
        sublist = [sp, train_y[count]]
        data.append(sublist)
        count += 1

    # PLS SELECTOR
    wave, ind = run_pls_selector(data, produce, num_waves)

    # HAGRID SELECTOR

    # Parameter lists for grid search
    crossover_rates = [0.25]
    mutation_rates = [0.05]
    tourny_sizes = [5]

    # Perform the grid search
    for cr in crossover_rates:
        for mr in mutation_rates:
            for ts in tourny_sizes:

                if not write:
                    print (produce + '-' + str(num_waves) + ': CR, MR, TS', cr, mr, ts)

                # Run GA wavelength selection
                _, ga_inds, ga_pop_scores = run_ga_selector(produce, num_waves, cr=cr,
                                                            mr=mr, tourny_size=ts)
                ga_pop = np.array(ga_pop_scores[0])
                ga_scores = ga_pop_scores[1]

                # Write the wavelengths selected by the GA
                if write:
                    ga_out = open('results/selected_wavelengths/' + produce + '_' + str(num_waves) + '_ga.txt', 'w')
                    for i, entry in enumerate(ga_pop):
                        ga_out.write(str(entry) + ',' + str(ga_scores[i]) + '\n')
                    ga_out.close()
