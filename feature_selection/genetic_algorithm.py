# Bad practice, but the warnings were getting annoying
import warnings
# Other imports
import numpy as np
from classifiers import knn, dtree
import random
import matplotlib.pyplot as plt
import time

warnings.filterwarnings("ignore")


class GA:
    """
    Genetic algorithm class, includes methods
    for crossover, mutation, and selection. If
    adaptive is set to True, the algorithm will
    attempt to find the optimal number of
    wavelengths as well
    """

    def __init__(self, mutation_rate, crossover_rate, classifier_type,
                 num_wavelengths, adaptive, data, population_size=100, iterations=50,
                 tourny_size=5, report_iou=False):
        self.classifier_type = classifier_type
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.data = data
        self.num_wavelengths = num_wavelengths
        self.population_size = population_size
        self.tourny_size = tourny_size
        self.iterations = iterations
        self.adaptive = adaptive
        self.population = self._initialize_population()
        self.report_iou = report_iou
        self.saved_scores = {}
        if report_iou:
            self.iou = []

    def _initialize_population(self):
        """
        Initialize the population to the specified
        population size where each member consists of
        a randomly selected set of wavelength indices
        whose length is dictated by num_wavelengths
        """

        total_wavelengths = len(self.data[0][0])
        pop = []

        for _ in range(self.population_size):
            if self.adaptive:
                member = np.random.choice(total_wavelengths, size=np.random.randint(3, 10))
            else:
                member = np.random.choice(total_wavelengths, size=self.num_wavelengths)
            pop.append(member)

        return pop

    def _create_classifier(self, data):
        """
        Given the specified classification type,
        create a classifier. Parameters for each
        classifier type are tuned here.
        """

        if self.classifier_type == 'knn':
            k = 11
            return knn(data, k)
        elif self.classifier_type == 'dtree':
            return dtree(data)

        else:
            print ("Invalid classifier selected!")

    def _score_members(self):
        """
        Score each member of the population
        """

        self.scores = []

        if self.adaptive:
            for member in self.population:
                input_data = self._create_data_subset(member)
                clf = self._create_classifier(input_data)
                score = clf.score() - (0.01 * len(member))

                # Apply a penalty for two wavelengths being within ~20nm of each other
                for i in range(len(member)):
                    for j in range(i + 1, len(member)):
                        if abs(member[i] - member[j]) <= 10:
                            score = score * 0.5

                self.scores.append(score)

        else:
            for member in self.population:
                input_data = self._create_data_subset(member)
                clf = self._create_classifier(input_data)
                score = clf.score()

                # Apply a penalty for two wavelengths being within ~20nm of each other
                # (this is used for the secondary fitness function)
                '''for i in range(len(member)):
                    for j in range(i+1, len(member)):
                        if abs(member[i] - member[j]) <= 10:
                            score = score * 0.5'''

                self.scores.append(score)

    def _create_data_subset(self, member):
        """
        Given a member of the population, transform
        the data to contain only the
        wavelengths specified by the member
        """

        d = [(np.take(self.data_trans[i][0], member),
              self.data_trans[i][1]) for i in range(len(self.data_trans))]

        return d

    def bin_wavelengths(self, data_in, wavelengths):
        """
        Given a set of wavelengths, bin the surrounding
        wavelengths (with bin width of ~30nm) and sum those
        bins
        """

        binned_spectra = []

        for d in data_in:

            binned_values = []

            for wave in wavelengths:
                start = max(wave - 7, 0)  # 7 indices is about 15 nanometers, which is half the bin width
                stop = min(len(data_in[0][0]), wave + 8)
                bin_sum = 0

                for i in range(start, stop):
                    bin_sum += d[0][i]

                binned_values.append(bin_sum)

            binned_spectra.append(np.array(binned_values))

        transformed = []
        for i, d in enumerate(data_in):
            transformed.append((binned_spectra[i], d[1]))

        return transformed

    def average_pairwise_distance(self):
        '''
        Calculate the average pairwise distance between each of the members
        of the population, as a metric for population diversity
        '''

        total_distance = 0
        sorted_pop = []
        for member in self.population:
            sorted_pop.append(np.sort(member))

        for i, member_1 in enumerate(sorted_pop):
            temp_dist = 0
            for j, member_2 in enumerate(sorted_pop):
                if i != j:
                    temp_dist += np.linalg.norm(member_1 - member_2)
            total_distance += temp_dist / (self.population_size - 1)

        return float(total_distance) / self.population_size

    def evolve(self):
        """
        For the specified number of iterations,
        complete selection, crossover, mutation,
        and replacement for the population
        """

        distances = []

        wave_indices = np.arange(146)

        # Uncomment following line to bin wavelengths before running GA
        # self.data_trans = self.bin_wavelengths(self.data, wave_indices)

        # Uncomment following line to use original data in GA
        self.data_trans = self.data

        start = time.time()
        count = 0
        for i in range(self.iterations):

            count += 1

            iter_start = time.time()
            if self.report_iou:
                new_iou = self.measure_iou()
                self.iou.append(new_iou)

            self._score_members()
            children = []

            # print ("Average fitness of round " + str(i) + ": ",sum(self.scores)/self.population_size, " max: ", max(self.scores))

            for _ in range(int(round(self.population_size / 2))):
                parent_1, parent_2 = self._tourny_select(self.tourny_size)
                child_1, child_2 = self._crossover(parent_1, parent_2)
                child_1 = self._mutate(child_1)
                child_2 = self._mutate(child_2)
                children.append(child_1)
                children.append(child_2)

            # For generational replacement
            self.population = children

            # For survival of the fittest
            """for child in children:
                self.population.append(child)
            self._cull_the_herd()"""

            # Average pairwise distance
            '''if i%1==0:
                dist = self.average_pairwise_distance()
                distances.append(dist)'''

            iter_stop = time.time()
            print ('Time for the iteration: ' + str(count), (iter_stop - iter_start))

        if self.report_iou:
            self.display_iou(print_to_term=True)

        stop = time.time()
        total_time = stop - start
        print ("    Total time, time per iteration: ", total_time, float(total_time) / self.iterations)
        print("     Average fitness: ", sum(self.scores) / self.population_size, " max: ", max(self.scores))
        # print (distances)

    def display_iou(self, print_to_term=False):
        """
        Plot the iou over time. If print_to_term
        is true, print the array of iou values
        to the terminal
        """

        if print_to_term:
            print ('-----------------------------')
            print ('IOU:')
            print (self.iou)
            print ('-----------------------------')

        plt.plot(self.iou)
        plt.xlabel('Generations', fontsize=14, labelpad=15)
        plt.ylabel('Intersection Over Union', fontsize=14, labelpad=15)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        plt.title("IOU Over Time", fontsize=20)
        plt.show()

    def _select_parents(self):
        """
        Fitness proportionate selection of
        two parents (DON'T USE THIS, USE
        TOURNAMENT SELECTION INSTEAD)
        """

        total_fitness = sum(self.scores)
        probabilities = np.array(self.scores) / total_fitness
        indices = np.arange(self.population_size)
        parents = np.random.choice(indices, size=2, replace=False, p=probabilities)

        return self.population[parents[0]], self.population[parents[1]]

    def _tourny_select(self, k):
        """
        Use tournament selection to select two parents
        with tournament size k
        """

        indices = np.arange(len(self.population))
        tourny_inds = np.random.choice(indices, k, replace=False)  # choose k random members

        # Calculate the probabilities for the tournament participants
        fitness = np.take(self.scores, tourny_inds)
        total_fitness = sum(fitness)
        probabilities = fitness / total_fitness

        # Select two winners
        winners = np.random.choice(tourny_inds, 2, replace=False, p=probabilities)

        return self.population[winners[0]], self.population[winners[1]]

    def _crossover(self, parent_1, parent_2):
        """
        Given two parents, perform binomial crossover
        if parents of same length, single point
        crossover otherwise
        """

        child_1 = []
        child_2 = []
        j_star = np.random.randint(len(parent_1))

        # Need to account for differing parent lengths
        # if running in adaptive mode, use single point
        # crossover
        if self.adaptive:

            cross_1 = np.random.randint(1, len(parent_1))
            cross_2 = np.random.randint(1, len(parent_2))
            first_half_1 = parent_1[:cross_1].tolist()
            first_half_2 = parent_2[:cross_2].tolist()
            second_half_1 = parent_1[cross_1:].tolist()
            second_half_2 = parent_2[cross_2:].tolist()
            child_1 = first_half_1 + second_half_2
            child_2 = first_half_2 + second_half_1

            if len(child_1) < 2:
                print ('one')
                print (cross_1, first_half_1, first_half_2)

            if len(child_2) < 2:
                print ('two')
                print (cross_2, second_half_1, second_half_2)

        # Case for the standard GA
        else:
            for j in range(len(parent_1)):
                rand_prob = np.random.rand()
                if rand_prob < self.crossover_rate or j == j_star:
                    child_1.append(parent_1[j])
                    child_2.append(parent_2[j])
                else:
                    child_1.append(parent_2[j])
                    child_2.append(parent_1[j])

        return np.array(child_1), np.array(child_2)

    def _mutate(self, child):
        """
        Mutate the offspring. Here, degree of mutation
        is a random integer over [-3, 3] that is added
        to the index/gene of a member of the population
        """

        total_wavelengths = len(self.data[0][0])

        for i in range(len(child)):
            rand_prob = np.random.rand()
            if rand_prob < self.mutation_rate:
                # Mutate gene
                mutation_amount = np.random.randint(-3, 4)
                child[i] = child[i] + mutation_amount

                # Make sure the index does not go out of bounds
                if child[i] > 149:
                    child[i] = 149
                if child[i] < 0:
                    child[i] = 0

        if self.adaptive:
            # g% of the time, either grow or shrink the
            # length of the child by one
            g = 0.1
            rand_prob = np.random.rand()
            if rand_prob < g:
                rand_prob = np.random.rand()

                # grow
                if rand_prob < 0.5:
                    new = np.random.choice(total_wavelengths, size=1)
                    child = np.append(child, new)

                # shrink (but not shorter than 2)
                elif len(child) > 2:
                    child = np.delete(child, random.randrange(len(child)))

        return child

    def _cull_the_herd(self):
        """
        Return the population to its original size
        by killing off the weakest (in terms of score)
        members
        """

        self._score_members()
        cutoff = -1 * self.population_size
        fittest_indices = np.argpartition(self.scores, cutoff)[cutoff:]
        self.population = np.take(self.population, fittest_indices, axis=0)
        self.population = self.population.tolist()

    def fetch_best_member(self):
        """
        Returns the best individual member of
        the population in terms of score
        """

        self._score_members()
        top_index = np.argmax(self.scores)
        # print ("Final score obtained: ", self.scores[top_index])
        return self.population[top_index]

    def measure_iou(self):
        """
        For the current population, measure the
        average intersection over union between
        the members of the population (sample the
        population, don't calculate for all
        possible pairs)
        """

        count = 0
        iou_total = 0

        for member in self.population:

            sort_mem = np.sort(member)

            # Calculate the iou of the member versus three other members
            compare = np.random.randint(low=0, high=len(self.population), size=3)

            for ind in range(len(self.population)):
                inter = np.intersect1d(sort_mem, np.sort(self.population[ind]))
                union = np.union1d(sort_mem, np.sort(self.population[ind]))
                new_iou = float(len(inter)) / len(union)

                count += 1
                iou_total += new_iou

        return round(iou_total / float(count), 3)
