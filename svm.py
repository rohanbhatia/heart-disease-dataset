from sklearn import svm

#used some ideas from http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
#linear poly rbf sigmoid

#linear kernel
def svm_linear(x_train, y_train, x_test, y_test):

	classifier = svm.SVC(kernel='linear')
	classifier.fit(x_train, y_train)
	
	print("Kernel: Linear")
	print("Performance: "  + str(classifier.score(x_test, y_test)))
	print("")

#polynomial kernel from degrees 2 to 5
def svm_poly(x_train, y_train, x_test, y_test):

	for d in [2, 3, 4, 5]:	
		
		classifier = svm.SVC(kernel='poly', degree=d, max_iter=400000)
		classifier.fit(x_train, y_train)
		
		print("Kernel: Polynomial")
		print("Degree: " + str(d))
		print("Performance: "  + str(classifier.score(x_test, y_test)))
		print("")

#radial basis function kernel
def svm_rbf(x_train, y_train, x_test, y_test):

	classifier = svm.SVC(kernel='rbf')
	classifier.fit(x_train, y_train)
	
	print("Kernel: Radial Basis Function")
	print("Performance: "  + str(classifier.score(x_test, y_test)))
	print("")

#sigmoid function kernel
def svm_sigmoid(x_train, y_train, x_test, y_test):

	classifier = svm.SVC(kernel='sigmoid')
	classifier.fit(x_train, y_train)
	
	print("Kernel: Sigmoid")
	print("Performance: "  + str(classifier.score(x_test, y_test)))
	print("")

