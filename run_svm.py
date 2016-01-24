import convert_data as cd

if __name__ == '__main__':
	X, T = cd.parse_data("data.txt")
	print(X)
	print(T)

