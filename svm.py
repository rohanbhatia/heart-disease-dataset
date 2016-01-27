from sklearn import svm

#used some ideas from http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
#linear poly rbf sigmoid precomputed

def svm_linear(x_train, y_train, x_test, y_test):

	classifier = svm.SVC(kernel='linear')
	classifier.fit(x_train, y_train)
	
	print("Kernel: Linear")
	print("Performance: "  + str(classifier.score(x_test, y_test)))
	print("")

def svm_poly(x_train, y_train, x_test, y_test):

	for d in [2, 3, 4, 5]:	
		
		classifier = svm.SVC(kernel='poly', degree=d, max_iter=400000)
		classifier.fit(x_train, y_train)
		
		print("Kernel: Polynomial")
		print("Degree: " + str(d))
		print("Performance: "  + str(classifier.score(x_test, y_test)))
		print("")

