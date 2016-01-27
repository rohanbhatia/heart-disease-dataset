import convert_data as cd
import numpy as np
from sklearn.utils.validation import check_array as check_arrays
from sklearn.cross_validation import train_test_split
import svm



if __name__ == '__main__':
	
	X, T = cd.parse_data("data.txt")
	
	x_train, x_test, y_train, y_test = train_test_split(X, T, test_size=0.40)
	
	svm.svm_linear(x_train, y_train, x_test, y_test)
	
	svm.svm_poly(x_train, y_train, x_test, y_test)

	svm.svm_rbf(x_train, y_train, x_test, y_test)

	svm.svm_sigmoid(x_train, y_train, x_test, y_test)

