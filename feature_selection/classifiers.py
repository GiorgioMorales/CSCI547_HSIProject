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

#Other imports
from abc import ABCMeta, abstractmethod
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier

class Classifier:
	"""
	Code for the Classifier abstract base class.
	The Classifier is the parent class for the
	various classification classes/algorithms.
	(Abstract base class is the same as an
	interface in other programming languages)
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

class knn(Classifier):
	"""
	k-nearest neighbors classifier.
	"""

	def __init__(self, data, k=3):
		super(knn, self).__init__(data)
		self.k = k
		self.model = None
		self.all_input, self.all_labels = self._split_inputs_outputs(self.data)

	def _fit(self, data):
		"""
		Fit the knn model.
		"""

		train_in, train_labels = self._split_inputs_outputs(data)
		clf = KNeighborsClassifier(n_neighbors=self.k)
		clf.fit(train_in, train_labels)

		return clf

	def score_one(self, test_data):
		"""
		Return the accuracy attained by the
		knn on the test data set.
		"""

		test_in, test_labels = self._split_inputs_outputs(test_data)
		correct = 0
		total = 0

		for i, test_input in enumerate(test_in):
			prediction = self.model.predict(test_input.reshape(1,-1))
			if prediction[0] == test_labels[i]:
				correct+=1
			total+=1
		return float(correct)/total

	def score(self):
		"""
		Use 10-fold CV to produce an average score
		"""

		splits = 10
		score = 0

		kf = KFold(n_splits=splits, shuffle=True)
		kf.get_n_splits(self.data)

		for train_ind, test_ind in kf.split(self.data):

			train = [self.data[ind] for ind in train_ind]
			test = [self.data[ind] for ind in test_ind]

			self.model = self._fit(train)
			temp_score = self.score_one(test)
			score += temp_score

		return score/float(splits)

class dtree(Classifier):
	"""
	Sklearn's decision tree implementation.
	"""

	def __init__(self, data):
		super(dtree, self).__init__(data)
		self.model = None
		self.all_input, self.all_labels = self._split_inputs_outputs(self.data)

	def _fit(self, data):
		"""
		Fit the decision tree model.
		"""

		train_in, train_labels = self._split_inputs_outputs(data)
		clf = DecisionTreeClassifier(min_samples_leaf=0.05)
		clf.fit(train_in, train_labels)

		return clf

	def score_one(self, test_data):
		"""
		Return the accuracy attained by the
		knn on the test data set.
		"""

		test_in, test_labels = self._split_inputs_outputs(test_data)
		correct = 0
		total = 0

		for i, test_input in enumerate(test_in):
			prediction = self.model.predict(test_input.reshape(1,-1))
			if prediction[0] == test_labels[i]:
				correct+=1
			total+=1
		return float(correct)/total

	def score(self):
		"""
		Use 10-fold CV to produce a score
		"""

		splits = 10
		score = 0

		kf = KFold(n_splits=splits, shuffle=True)
		kf.get_n_splits(self.data)

		for train_ind, test_ind in kf.split(self.data):

			train = [self.data[ind] for ind in train_ind]
			test = [self.data[ind] for ind in test_ind]

			self.model = self._fit(train)
			temp_score = self.score_one(test)
			score += temp_score

		return score/float(splits)

class Logistic(Classifier):
	"""
	Logistic regression classifier
	"""

	def __init__(self, data, inputs=None, outputs=None):
		super(Logistic, self).__init__(data)
		self.inputs, self.labels = self._split_inputs_outputs(data)
		self.model = self._fit(inputs, labels)

	def _fit(self):
		"""
		Fit the logistic regression model.
		"""

		clf = LogisticRegression()
		clf.fit(inputs, labels)

		return clf

	def score(self, test_data):
		"""
		Return the score obtained on the
		test data set
		"""

		ins, outs = self._split_inputs_outputs(test_data)
		return self.model.score(ins, outs)

class FFNN(Classifier):
	"""
	Feed-forward neural network
	"""

	def __init__(self, data):
		super(FFNN, self).__init__(data)
		self.inputs, self.labels = self._split_inputs_outputs(data)

	def train(self):
		"""
		Train the neural network using Adam optimizer
		"""

		input_size = len(self.inputs[0])
		output_size =  len(set(self.labels))
		hidden_size_1 = 15
		hidden_size_2 = 15

		# One hot encode the labels
		encoder = LabelEncoder()
		encoder.fit(self.labels)
		enc_labels = encoder.transform(self.labels)
		enc_labels = np_utils.to_categorical(enc_labels)

		# Create the MLP
		model = Sequential()
		model.add(Dense(hidden_size_1, activation='relu', input_dim=input_size))
		model.add(Dense(hidden_size_2, activation='relu'))
		model.add(Dense(output_size, activation='softmax'))

		# Compile model with optimizer and loss function
		model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

		# Train the model
		model.fit(self.inputs, enc_labels, steps_per_epoch=1000, epochs=20, verbose=2)

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


assert issubclass(knn, Classifier)
#Classifier.register(knn)
