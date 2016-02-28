import convert_data as cd
import numpy as np
from sklearn.utils.validation import check_array as check_arrays
from sklearn.cross_validation import train_test_split
import svm
import logistic_regression as lr



if __name__ == '__main__':
	
	#Get training examples and targets
	X, T = cd.parse_data("data.txt")
	
	#Split data into training and testing sets
	x_train, x_test, y_train, y_test = train_test_split(X, T, test_size=0.40)
	
	#Run SVM with a linear kernel
	#svm.svm_linear(x_train, y_train, x_test, y_test)
	
	#Run SVM with a polynomial kernel
	#svm.svm_poly(x_train, y_train, x_test, y_test)

	#Run SVM with a radial basis function kernel
	#svm.svm_rbf(x_train, y_train, x_test, y_test)

	#Run SVM with a sigmoid kernel
	#svm.svm_sigmoid(x_train, y_train, x_test, y_test)

	#Run Logistic Regression
	lr.run_regression(True, 30, "Logistic Regression", x_train, y_train, x_test, y_test)

	#Run Scaled Logistic Regression
	lr.run_regression(True, 30, "Scaled Logistic Regression", x_train, y_train, x_test, y_test)

	#Run Log-transform Logistic Regression
	lr.run_regression(True, 30, "Log-transform Logistic Regression", x_train, y_train, x_test, y_test)

	#Run Binary-transform Logistic Regression
	lr.run_regression(True, 30, "Binary-transform Logistic Regression", x_train, y_train, x_test, y_test)




