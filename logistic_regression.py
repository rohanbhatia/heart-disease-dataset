import numpy as np
import math
import copy
from sklearn import linear_model, cross_validation, preprocessing

def parse_data(input_file):
	#INPUT: An input file containing the data
	#OUTPUT: Returns a vector of inputs and a vector of targets

	inputs = list()
	targets = list()

	f = open(input_file)

	#parse each line
	gl_count = 0
	for line in f:	
		
		if (gl_count > 1):

			example = line.split(",")
		
			count = 0
			while count < 58:
				example[count] = float(example[count])
				count += 1

			#add data example and it's target to the main input/target arrays
			inputs.append(example[0:57])
			targets.append(example[57])

		else:
			gl_count += 1
	
	f.close()

	X = np.array(inputs)
	T = np.array(targets)

	return X, T


def logistic_regression(x_train, t_train, x_test, t_test):

	_5_fold = cross_validation.KFold(
		len(x_train), 
		n_folds=5, 
		shuffle=False, 
		random_state=None)

	max_iterations = [100]

	results = list()
	
	for i in max_iterations:

		classifier = linear_model.LogisticRegressionCV(
		Cs=10, 
		fit_intercept=True, 
		cv=_5_fold, 
		dual=False, 
		penalty='l2', 
		scoring=None, 
		solver='liblinear', 
		tol=0.0001, 
		max_iter=i, 
		class_weight=None, 
		n_jobs=1, 
		verbose=0, 
		refit=True, 
		intercept_scaling=1.0, 
		multi_class='ovr', 
		random_state=None)


		classifier.fit(
			x_train, 
			t_train, 
			sample_weight=None)

		train_result = 100*(1-classifier.score(
			x_train,
			t_train,
			sample_weight=None))

		test_result = 100*(1-classifier.score(
			x_test, 
			t_test, 
			sample_weight=None))

		reg = math.pow(classifier.C_, -1)

		results.append([train_result, test_result, reg, classifier])

	return results[0]


def scaled_logistic_regression(x_train, t_train, x_test, t_test):

	 x_train_new = preprocessing.scale(x_train)

	 x_test_new = preprocessing.scale(x_test)
	 
	 return logistic_regression(x_train_new, t_train, x_test_new, t_test)

def log_transform_logistic_regression(x_train, t_train, x_test, t_test):

	x_train_new = copy.deepcopy(x_train)

	for row in x_train_new:
		for column in row:
			column = math.log1p(column)

	x_test_new = copy.deepcopy(x_test)

	for row in x_test_new:
		for column in row:
			column = math.log1p(column)

	return logistic_regression(x_train_new, t_train, x_test_new, t_test)

def binary_transform_logistic_regression(x_train, t_train, x_test, t_test):

	x_train_new = copy.deepcopy(x_train)

	for row in x_train_new:
		for column in row:
			if column != 0:
				column = 1
	
	x_test_new = copy.deepcopy(x_test)

	for row in x_test_new:
		for column in row:
			if column != 0:
				column = 1

	return logistic_regression(x_train_new, t_train, x_test_new, t_test)

def run_regression(verbose, n_trials, algo_name, x_train, t_train, x_test, t_test):

	count = 0
	best_rate = 100.00
	reg = 0
	total_train = 0
	total_test = 0

	if verbose:

		print("train\t\ttest\t\treg")

	while count < n_trials:

		if algo_name == "Logistic Regression":

			results = logistic_regression(x_train, t_train, x_test, t_test)

		elif algo_name == "Scaled Logistic Regression":

			results = scaled_logistic_regression(x_train, t_train, x_test, t_test)

		elif algo_name ==  "Log-transform Logistic Regression":

			results = log_transform_logistic_regression(x_train, t_train, x_test, t_test)

		elif algo_name == "Binary-transform Logistic Regression":

			results = binary_transform_logistic_regression(x_train, t_train, x_test, t_test)

		if verbose:

			print(str(round(results[0], 2)) + "%\t\t" + str(round(results[1], 2)) + "%\t\t" + str(round(results[2], 8)))

		total_train += results[0]
		total_test += results[1]

		if results[1] <= best_rate:
			best_rate = results[1]
			reg = results[2]

		count += 1

	print("AVERAGE TRAINING ERROR: " + str(round((total_train/count), 3)) + "%")
	print("AVERAGE TESTING ERROR: " + str(round((total_test/count), 3)) + "%")
	print("BEST REG VALUE: " + str(reg))
	print("NUMBER OF TRIALS: " + str(n_trials))


def get_best_model_features(x_train, t_train, x_test, t_test):

	results = logistic_regression(x_train, t_train, x_test, t_test)

	classifier = results[3]

	highest_features = list()
	lowest_features = list()

	count = 0

	b = copy.deepcopy(classifier.coef_)

	c = list()

	for g in b[0]:

		#c.append(copy.deepcopy(abs(g)))
		c.append(copy.deepcopy(g))


	a = np.array(c)
	b = np.array(c)

	while count < 5:

		i = np.argmax(a)
		highest_features.append(i)
		a[i] = 0

		j = np.argmin(b)
		lowest_features.append(j)
		b[j] = 0

		count += 1

	print("Most positive features: " + str(highest_features))
	print("Most negative features: " + str(lowest_features))


if __name__ == '__main__':
	
	x_train, t_train = parse_data("spambase.train.txt")
	x_test, t_test = parse_data("spambase.test.txt")

	# print("\nLogistic Regression")
	# run_regression(True, 30, "Logistic Regression", x_train, t_train, x_test, t_test)

	# print("\nScaled Logistic Regression")
	# run_regression(True, 30, "Scaled Logistic Regression", x_train, t_train, x_test, t_test)

	# print("\nLog-transform Logistic Regression")
	# run_regression(True, 30, "Log-transform Logistic Regression", x_train, t_train, x_test, t_test)
	
	# print("\nBinary-transform Logistic Regression")
	# run_regression(True, 30, "Binary-transform Logistic Regression", x_train, t_train, x_test, t_test)

	print("\nBest model's highest features")
	get_best_model_features(x_train, t_train, x_test, t_test)






	