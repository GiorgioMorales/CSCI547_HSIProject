#############################################
#                                           #
# Code for various different classification #
# algorithms to be used in conjunction      #
# with the wavelength selection algorithms. #
#                                           #
#############################################

# Bad practice, but the warnings were getting annoying
import warnings

warnings.filterwarnings("ignore")

# Other imports
from abc import ABCMeta, abstractmethod
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras import optimizers, backend
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold


class Classifier:
    """
	Code for the Classifier abstract base class.
	The Classifier is the parent class for the
	various classification classes/algorithms.
	"""

    __metaclass__ = ABCMeta

    def __init__(self, data):
        self.data = data
        self.model = None

    def _split_inputs_outputs(self, data):
        """
		Split out the data set into its
		inputs and its labels.
		"""

        inputs = []
        outputs = []

        for point in data:
            inputs.append(point[0])
            outputs.append(point[1])

        return np.array(inputs), np.array(outputs)

    @abstractmethod
    def score(self, test_data):
        """
		Method to classify a test data set
		and return the score in terms of
		accuracy.
		"""

        pass


class MLP(Classifier):
    """
	Feed-forward neural network
	"""

    def __init__(self, data, lr=0.001, hidden=5):
        super(MLP, self).__init__(data)
        self.inputs, self.labels = self._split_inputs_outputs(data)
        self.hidden = hidden
        self.lr = lr

    def train(self):
        """
		Train the neural network using Adam optimizer
		"""

        input_size = len(self.inputs[0])
        output_size = len(set(self.labels))
        hidden_size_1 = self.hidden
        # hidden_size_2 = 15

        # One hot encode the labels
        encoder = LabelEncoder()
        encoder.fit(self.labels)
        enc_labels = encoder.transform(self.labels)
        enc_labels = np_utils.to_categorical(enc_labels)

        # Create the MLP
        model = Sequential()
        model.add(Dense(hidden_size_1, activation='relu', input_dim=input_size))
        # model.add(Dense(hidden_size_2, activation='relu'))
        model.add(Dense(output_size, activation='sigmoid'))

        # Adam optimizer
        adam = optimizers.adam(lr=self.lr)

        # Compile model with optimizer and loss function
        model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

        # Train the model
        model.fit(self.inputs, enc_labels, steps_per_epoch=500, epochs=30)
        self.model = model

    def score(self, test_data):
        """
		Return the accuracy attained by the neural network
		on the test set
		"""

        ins, outs = self._split_inputs_outputs(test_data)

        # One hot encode the input/labels
        encoder = LabelEncoder()
        encoder.fit(outs)
        enc_labels = encoder.transform(outs)
        enc_labels = np_utils.to_categorical(enc_labels)

        _, score = self.model.evaluate(ins, enc_labels, verbose=2)

        return score

    def confusion_matrix(self, test_data, matrix=None):
        '''
		Produce a confusion matrix for the test data. If
		no matrix is provided, create a new matrix, otherwise
		update the provided matrix
		'''

        ins, outs = self._split_inputs_outputs(test_data)

        # One hot encode the input/labels
        encoder = LabelEncoder()
        encoder.fit(outs)
        enc_labels = encoder.transform(outs)
        enc_labels = np_utils.to_categorical(enc_labels)
        single_labels = [int(np.argmax(enc_labels[i])) for i in range(len(enc_labels))]

        # If no matrix provided, create a new one, otherwise update existing one
        try:
            if matrix == None:
                n = int(np.add(np.max(single_labels), 1))
                conf_mat = np.zeros((n, n))
        except ValueError:
            conf_mat = matrix

        # Get the predictions of the model
        raw_predictions = self.model.predict(ins)
        predictions = [np.argmax(p) for p in raw_predictions]

        # update the confusion matrix
        for i in range(len(predictions)):
            true = single_labels[i]
            predicted = int(predictions[i])
            conf_mat[true][predicted] += 1

        return conf_mat
