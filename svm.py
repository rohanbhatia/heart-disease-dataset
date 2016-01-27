from sklearn import svm

#used some ideas from http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

#linear kernel
def svm_linear(x_train, y_train, x_test, y_test):

	classifier = svm.SVC(
		C=1.0, 
		kernel='linear', 
		probability=False, 
		shrinking=True, 
		tol=1e-3, 
		verbose=False, 
		max_iter=-1, 
		decision_function_shape=None,
		random_state=None)

	classifier.fit(x_train, y_train)
	
	print("Kernel: Linear")
	print("Performance: "  + str(classifier.score(x_test, y_test)))
	print("")

#polynomial kernel from degrees 2 to 5
def svm_poly(x_train, y_train, x_test, y_test):

	for d in [2, 3, 4, 5]:	

		classifier = svm.SVC(
			C=1.0,
			kernel='poly', 
			degree=d,
			gamma='auto',
			coef0=0.0,
			probability=False,
			shrinking=True,
			tol=1e-3,
			verbose=False,
			max_iter=400000,
			decision_function_shape=None,
			random_state=None)

		classifier.fit(x_train, y_train)
		
		print("Kernel: Polynomial")
		print("Degree: " + str(d))
		print("Performance: "  + str(classifier.score(x_test, y_test)))
		print("")

#radial basis function kernel
def svm_rbf(x_train, y_train, x_test, y_test):

	classifier = svm.SVC(
		C=1.0,
		kernel='rbf',
		gamma='auto',
		probability=False,
		shrinking=True,
		tol=1e-3,
		verbose=False,
		max_iter=-1,
		decision_function_shape=None,
		random_state=None)

	classifier.fit(x_train, y_train)
	
	print("Kernel: Radial Basis Function")
	print("Performance: "  + str(classifier.score(x_test, y_test)))
	print("")

#sigmoid function kernel
def svm_sigmoid(x_train, y_train, x_test, y_test):

	classifier = svm.SVC(
		C=1.0,
		kernel='sigmoid',
		gamma='auto',
		coef0=0.0,
		probability=False,
		shrinking=True,
		tol=1e-3,
		verbose=False,
		max_iter=-1,
		decision_function_shape=None,
		random_state=None)
	classifier.fit(x_train, y_train)
	
	print("Kernel: Sigmoid")
	print("Performance: "  + str(classifier.score(x_test, y_test)))
	print("")

