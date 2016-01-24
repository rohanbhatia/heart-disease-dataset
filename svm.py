from sklearn import svm

def train_svm(x_train, y_train, x_test, y_test, kernel=None):

	classifier = svm.SVC()
	classifier.fit(x_train, y_train)

	m = len(x_test)
	correct = 0
	count = 0

	while(count < m):

		y = classifier.predict(x_test[count])

		print(y)

		count += 1



