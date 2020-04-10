from abc import ABCMeta, abstractmethod
from sklearn.cross_decomposition import PLSRegression
from random import shuffle
import numpy as np
import pickle
from genetic_algorithm import GA
from classifiers import knn, Logistic
import utils


class Selector:
    """
    Code for the Selector abstract base class. The
    Selector is the parent class for the various
    wavelength selection classes/algorithms.
    """

    __metaclass__ = ABCMeta

    def __init__(self, data, num_wavelengths):
        shuffle(data)
        self.data = data
        self.num_wavelengths = num_wavelengths
        self.all_wavelengths = self._set_wavelengths()

    def _set_wavelengths(self):
        """
        Include all the available wavelengths from
        the imager.
        """

        waves = utils.imager_waves
        return waves

    def _partition_data(self):
        """
        Split the data into train and test
        """

        split = int(round(9 * len(self.data) / 10))
        return self.data[:split], self.data[split:]

    def split_input_output(self, data):
        """
        Split the dataset into inputs and outputs
        """

        inputs = np.array([datum[0] for datum in data])
        labels = np.array([datum[1] for datum in data])

        return inputs, labels

    @abstractmethod
    def find_wavelengths(self):
        """
        Method to find the "optimal" wavelengths
        as determined by each of the algorithms.
        """

        pass

    @abstractmethod
    def test_model(self):
        """
        Method to test the classification model
        and return the accuracy attained without
        actually selecting wavelenghts.
        """

        pass


class SPASelector(Selector):
    """
    Successive Projections Algorithm for
    wavelength selection (unused in the final
    project and untested for a while. May or
    may not work as expected)
    """

    def __init__(self, data, num_wavelengths):
        super(SPASelector, self).__init__(data, num_wavelengths)
        self.selected_wavelengths = []
        self.remaining_wavelenths = list(xrange(len(data[0][0])))

    def test_model(self):
        """
        Test the successive projection algorithm
        without finding specfic wavelengths
        """

        # Split data into train and test
        train, test = self._partition_data()
        train_in, train_labels = self.split_input_output(train)
        test_in, test_labels = self.split_input_output(test)

        # Logistic regression to test the selected wavelengths
        lr = Logistic("dummy", train_in, train_labels)
        original_score = lr.score(test_in, test_labels)

        # Get the first wavelength using PLSDA Selector
        # (the first wavelength must be known a priori)
        plsda = PLSDASelector(self.data, self.num_wavelengths)
        first_wavelength_ind, first_wavelength = plsda.find_wavelengths(True)
        self.selected_wavelengths.append(first_wavelength_ind)
        self.remaining_wavelenths.remove(first_wavelength_ind)

        # Select the remaining wavelenths
        for _ in range(self.num_wavelengths - 1):
            self._select_next(train_in)

        new_train_in = self._create_data_subset(train_in)
        new_test_in = self._create_data_subset(test_in)
        lr2 = Logistic("dummy", new_train_in, train_labels)
        new_score = lr2.score(new_test_in, test_labels)

        print (self.selected_wavelengths)

        print ("Score before selecting wavelengths: ", original_score)
        print ("Score after selecting (" + str(self.num_wavelengths) + ") wavelengths: ", new_score)

    def find_wavelengths(self):
        """
        Return the optimal wavelengths found
        """

        # Split data into train and test
        train, test = self._partition_data()
        train_in, train_labels = self.split_input_output(train)
        test_in, test_labels = self.split_input_output(test)

        # Get the first wavelength using PLSDA Selector
        # (the first wavelength must be known a priori)
        plsda = PLSDASelector(self.data, self.num_wavelengths)
        first_wavelength_ind, first_wavelength = plsda.find_wavelengths(True)
        self.selected_wavelengths.append(first_wavelength_ind)
        self.remaining_wavelenths.remove(first_wavelength_ind)

        # Select the remaining wavelenths
        for _ in range(self.num_wavelengths - 1):
            self._select_next(train_in)

        self.selected_wavelengths.sort()

        optimal_wavelengths = []
        for index in self.selected_wavelengths:
            optimal_wavelengths.append(self.all_wavelengths[index])

        return np.array(optimal_wavelengths), np.array(self.selected_wavelengths)

    def _select_next(self, inputs):
        """
        Select the next wavelength using the projection
        operator, add it to the list of selected
        wavelengths and remove from list of remaining
        """

        # We need to access the input data by column later
        input_columns = inputs.transpose()

        max_score = float("-inf")
        best_ind = None

        # iterate over all remaining wavelengths
        # and keep track of the argmax
        for i in self.remaining_wavelenths:
            Px_i = self._project(input_columns[i], input_columns[self.selected_wavelengths[-1]])
            temp_score = np.linalg.norm(Px_i)

            if temp_score > max_score:
                max_score = temp_score
                best_ind = i

        self.selected_wavelengths.append(best_ind)
        self.remaining_wavelenths.remove(best_ind)

    def _create_data_subset(self, inputs):
        """
        Produce a new data set with only the
        selected wavelengths
        """

        new_inputs = []

        for i, item in enumerate(inputs):
            reduced_set = np.take(item, self.selected_wavelengths)
            new_inputs.append(reduced_set)

        return new_inputs

    def _project(self, input_i, input_k):
        """
        Return the orthogonal projection of x_i
        relative to the last selected wavelength k
        """

        x_i = input_i.reshape((input_i.size, 1))
        x_k = input_k.reshape((input_k.size, 1))

        # The three terms for the projection
        term_1 = np.matmul(x_k, x_i.transpose())
        term_2 = x_k
        term_3 = np.linalg.inv(np.matmul(x_k.transpose(), x_k))

        # Multiply the terms together
        temp = np.matmul(term_1, term_2)
        final = np.matmul(temp, term_3)

        return final


class PLSDASelector(Selector):
    """
    Partial least squares discriminant analysis
    for wavelength selection.
    """

    def __init__(self, data, num_wavelengths, iterations=500):
        super(PLSDASelector, self).__init__(data, num_wavelengths)
        self.iterations = iterations

    def test_model(self):
        """
        Split data into train and test to
        evaluate the performance of the model.
        """

        # Split data into train and test
        train, test = self._partition_data()

        # Separate out the inputs and label output,
        # then encode 'fresh' as 1 and 'shelf' as 0
        train_inputs = np.array([datum[0] for datum in train])
        test_inputs = np.array([datum[0] for datum in test])
        train_labels = [datum[-1] == 'fresh' for datum in train]
        train_labels = np.array(train_labels, dtype=int)
        test_labels = [datum[-1] == 'fresh' for datum in test]
        test_labels = np.array(test_labels, dtype=int)

        # Use pls regression to perform PLS-DA
        plsr = PLSRegression(n_components=10, scale=False)
        plsr.fit(train_inputs, train_labels)
        mypred = plsr.predict(test_inputs)

        total = 0
        right = 0

        # Count up the correct predictions
        for i, pred in enumerate(mypred):
            if pred[0] < 0.5:
                if test_labels[i] == 0:
                    right += 1
                total += 1
            else:
                if test_labels[i] == 1:
                    right += 1
                total += 1

        print ("percent correct: ", float(right) / total)

    def find_wavelengths(self, top_one=False):
        """
        Main call to return the wavelengths identified
        as being the most "informative" by PLS-DA.
        """

        # Separate out the inputs and label output,
        # then encode 'fresh' as 1 and 'shelf' as 0
        inputs = np.array([datum[0] for datum in self.data])
        labels = [datum[-1] == 0 for datum in self.data]
        labels = np.array(labels, dtype=int)

        # Use pls regression to perform PLS-DA
        plsr = PLSRegression(n_components=10, scale=False)
        plsr.fit(inputs, labels)
        mypred = plsr.predict(inputs)

        # Find the indices for the coefficients of the
        # regression with the highest absolute values
        coefficients = np.absolute(np.array([coef[0] for coef in plsr.coef_]))
        top_indices = np.argpartition(coefficients,
                                      (-1 * self.num_wavelengths))[(-1 * self.num_wavelengths):]
        optimal_wavelengths = []

        # Other algorithms need the first wavelength a
        # priori, this top_one flag allows the method
        # to return the most informative wavelength found
        if top_one:
            return top_indices[0], self.all_wavelengths[top_indices[0]]

        top_indices = np.sort(top_indices)

        for i in top_indices:
            optimal_wavelengths.append(self.all_wavelengths[i])
        optimal_wavelengths.sort()

        return np.array(optimal_wavelengths), top_indices


class GASelector(Selector):
    """
    Genetic algorithm for wavelength selection.
    Must pass the classifier as an argument, and
    the classification accuracy of the specified
    classifier serves as the objective function.
    """

    def __init__(self, data, num_wavelengths, classifier_type='knn', mutation_rate=0.3,
                 crossover_rate=0.35, tourny_size=5):
        super(GASelector, self).__init__(data, num_wavelengths)
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.gen_alg = None
        self.classifier_type = classifier_type
        self.tourny_size = tourny_size
        if num_wavelengths == 0:
            self.num_wavelengths = 6
            self.adaptive = True
        else:
            self.adaptive = False
        # self.classifier = None

    def find_wavelengths(self):
        """
        Return the optimal wavelengths (and their
        associated indices) found by the genetic
        algorithm.
        """

        # Split data into train and test
        train, test = self._partition_data()

        # Create the GA based on the classifier
        self.gen_alg = GA(self.mutation_rate, self.crossover_rate, self.classifier_type,
                          self.num_wavelengths, self.adaptive, self.data, population_size=1000,
                          iterations=100, tourny_size=self.tourny_size, report_iou=False)

        self.gen_alg.evolve()
        indices = self.gen_alg.fetch_best_member()
        indices = np.sort(indices)

        optimal_wavelengths = []
        for index in indices:
            optimal_wavelengths.append(self.all_wavelengths[index])

        return np.array(optimal_wavelengths), indices

    def test_model(self):
        """
        Report the accuracy attained by the GA
        on the dataset without actually selecting
        wavelengths.
        """

        # Split data into train and test
        train, test = self._partition_data()

        # Create the GA based on the classifier
        self.gen_alg = GA(self.mutation_rate, self.crossover_rate, self.classifier_type,
                          train, test, self.num_wavelengths, population_size=100, iterations=50)

        self.gen_alg.evolve()


assert issubclass(PLSDASelector, Selector)
assert issubclass(GASelector, Selector)
assert issubclass(SPASelector, Selector)
